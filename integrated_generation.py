"""
Fast-dLLM × LongLLaDA 統合生成機能
Fast-dLLMの正しいデュアルキャッシュとLongLLaDAのRoPEスケーリングを組み合わせた高速長文生成
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple, List
import time
import sys
import os
import glob

# Fast-dLLMのパスを追加
print(f"🔍 現在のディレクトリ: {os.path.dirname(os.path.abspath(__file__))}")

# Colab環境での診断
possible_paths = [
    # 現在のディレクトリからの相対パス
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'Fast-dLLM', 'llada', 'model'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'Fast-dLLM', 'llada'),
    # 直接のFast-dLLMディレクトリ
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fast-dLLM'),
    # 親ディレクトリからの検索
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'Fast-dLLM', 'llada', 'model'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'Fast-dLLM', 'llada'),
]

print("📁 ディレクトリ診断:")
for path in possible_paths:
    exists = os.path.exists(path)
    print(f"  {path}: {'✅' if exists else '❌'}")
    if exists and os.path.isdir(path):
        try:
            contents = os.listdir(path)[:5]  # 最初の5項目のみ
            print(f"    内容: {contents}")
        except:
            pass

# modeling_llada.pyファイルを探す
print("\n🔍 modeling_llada.py を検索中...")
for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
    if 'modeling_llada.py' in files:
        print(f"✅ 見つかりました: {root}")
        model_path = root
        generate_path = os.path.dirname(
            root) if root.endswith('model') else root
        break
else:
    # フォールバック: project layout情報から推測
    print("🔍 プロジェクトレイアウトから推測...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(
        __file__)), 'Fast-dLLM', 'Fast-dLLM', 'llada', 'model')
    generate_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'Fast-dLLM', 'Fast-dLLM', 'llada')

print(f"📍 使用するパス:")
print(f"  - モデルパス: {model_path}")
print(f"  - 生成パス: {generate_path}")

sys.path.insert(0, model_path)
sys.path.insert(0, generate_path)

try:
    from modeling_llada import LLaDAModelLM
    from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
    print("✅ Fast-dLLMの正しい実装を読み込みました")
except ImportError as e:
    print(f"⚠️  Fast-dLLMの実装が見つかりません: {e}")
    # フォールバック: Hugging Face Hub のAutoModelForCausalLMを使用
    print("⚠️  AutoModelForCausalLMを使用します（キャッシュ機能制限あり）")
    LLaDAModelLM = AutoModelForCausalLM
    generate = None
    generate_with_prefix_cache = None
    generate_with_dual_cache = None


def add_gumbel_noise(logits, temperature):
    """
    Gumbel Max サンプリング（Fast-dLLMの実装）
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    uniform = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
    return logits + gumbel * temperature


def get_num_transfer_tokens(mask_index, steps):
    """
    各ステップで転送するトークン数を事前計算（Fast-dLLMの実装）
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(
        0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    """
    信頼度ベースの並列デコーディング（Fast-dLLMの実装）
    """
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False

    return x0, transfer_index


@torch.no_grad()
def generate_fast_long_dual_cache(model, prompt, steps=128, gen_length=128, block_length=32,
                                  temperature=0., remasking='low_confidence', mask_id=126336,
                                  threshold=None, scaling_factor=1):
    """
    Fast-dLLM × LongLLaDA 統合生成関数（デュアルキャッシュ使用）

    Args:
        model: LLaDAモデル
        prompt: 入力プロンプト (1, L)
        steps: 拡散ステップ数
        gen_length: 生成長
        block_length: ブロックサイズ
        temperature: サンプリング温度
        remasking: リマスキング戦略
        mask_id: マスクトークンID
        threshold: 信頼度閾値
        scaling_factor: RoPEスケーリング係数

    Returns:
        tuple: (生成結果, NFE数, メトリクス)
    """
    start_time = time.time()
    device = model.device

    # RoPEスケーリング適用（LongLLaDA）
    original_theta = None
    if scaling_factor > 1 and hasattr(model.config, 'rope_theta'):
        original_theta = model.config.rope_theta
        model.config.rope_theta = original_theta * scaling_factor
        print(f"🔧 RoPE θ スケーリング: {original_theta} → {model.config.rope_theta}")

    # Fast-dLLMの正しい実装が利用可能な場合はそれを使用
    if generate_with_dual_cache is not None:
        print("🚀 Fast-dLLMの正しいデュアルキャッシュ実装を使用")
        outputs, nfe = generate_with_dual_cache(
            model, prompt, steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=temperature, remasking=remasking, mask_id=mask_id, threshold=threshold
        )

        # RoPEスケーリングを元に戻す
        if original_theta is not None:
            model.config.rope_theta = original_theta

        total_time = time.time() - start_time
        metrics = {
            'total_time': total_time,
            'tokens_per_second': gen_length / total_time if total_time > 0 else 0,
            'nfe': nfe,
            'cache_hits': nfe - 1,  # 推定値
            'total_tokens_generated': gen_length,
            'blocks_processed': gen_length // block_length,
            'scaling_factor_used': scaling_factor
        }
        return outputs, nfe, metrics

    # フォールバック: キャッシュなし実装
    print("⚠️  デュアルキャッシュ利用不可、キャッシュなし実装を使用")
    return generate_no_cache(
        model, prompt, steps, gen_length, block_length,
        temperature, remasking, mask_id, threshold, scaling_factor
    )


@torch.no_grad()
def generate_fast_long_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=32,
                                    temperature=0., remasking='low_confidence', mask_id=126336,
                                    threshold=None, scaling_factor=1):
    """
    Fast-dLLM × LongLLaDA 統合生成関数（プレフィックスキャッシュ使用）
    """
    start_time = time.time()

    # RoPEスケーリング適用
    original_theta = None
    if scaling_factor > 1 and hasattr(model.config, 'rope_theta'):
        original_theta = model.config.rope_theta
        model.config.rope_theta = original_theta * scaling_factor
        print(f"🔧 RoPE θ スケーリング: {original_theta} → {model.config.rope_theta}")

    # Fast-dLLMの正しい実装が利用可能な場合はそれを使用
    if generate_with_prefix_cache is not None:
        print("🚀 Fast-dLLMの正しいプレフィックスキャッシュ実装を使用")
        outputs, nfe = generate_with_prefix_cache(
            model, prompt, steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=temperature, remasking=remasking, mask_id=mask_id, threshold=threshold
        )

        # RoPEスケーリングを元に戻す
        if original_theta is not None:
            model.config.rope_theta = original_theta

        total_time = time.time() - start_time
        metrics = {
            'total_time': total_time,
            'tokens_per_second': gen_length / total_time if total_time > 0 else 0,
            'nfe': nfe,
            'cache_hits': nfe - 1,  # 推定値
            'total_tokens_generated': gen_length,
            'blocks_processed': gen_length // block_length,
            'scaling_factor_used': scaling_factor
        }
        return outputs, nfe, metrics

    # フォールバック: キャッシュなし実装
    print("⚠️  プレフィックスキャッシュ利用不可、キャッシュなし実装を使用")
    return generate_no_cache(
        model, prompt, steps, gen_length, block_length,
        temperature, remasking, mask_id, threshold, scaling_factor
    )


def generate_fast_long(model, prompt, steps=128, gen_length=128, block_length=32,
                       temperature=0., remasking='low_confidence', mask_id=126336,
                       threshold=None, use_cache=True, scaling_factor=1, dual_cache=True):
    """
    統合生成関数（キャッシュ方式を選択可能）
    """
    # Fast-dLLMの正しい実装が利用可能な場合はそれを優先使用
    if use_cache and generate is not None:
        if dual_cache and generate_with_dual_cache is not None:
            print("🎯 Fast-dLLM デュアルキャッシュ実装を使用")
            return generate_fast_long_dual_cache(
                model, prompt, steps, gen_length, block_length,
                temperature, remasking, mask_id, threshold, scaling_factor
            )
        elif generate_with_prefix_cache is not None:
            print("🎯 Fast-dLLM プレフィックスキャッシュ実装を使用")
            return generate_fast_long_prefix_cache(
                model, prompt, steps, gen_length, block_length,
                temperature, remasking, mask_id, threshold, scaling_factor
            )

    # フォールバック: キャッシュなし実装（最も安全）
    print("🎯 キャッシュなし実装を使用（安全モード）")
    return generate_no_cache(
        model, prompt, steps, gen_length, block_length,
        temperature, remasking, mask_id, threshold, scaling_factor
    )


@torch.no_grad()
def generate_no_cache(model, prompt, steps=128, gen_length=128, block_length=32,
                      temperature=0., remasking='low_confidence', mask_id=126336,
                      threshold=None, scaling_factor=1):
    """
    キャッシュなし生成（Fast-dLLMの標準実装ベース）
    """
    start_time = time.time()

    # RoPEスケーリング適用
    original_theta = None
    if scaling_factor > 1 and hasattr(model.config, 'rope_theta'):
        original_theta = model.config.rope_theta
        model.config.rope_theta = original_theta * scaling_factor
        print(f"🔧 RoPE θ スケーリング: {original_theta} → {model.config.rope_theta}")

    # Fast-dLLMの正しい実装が利用可能な場合はそれを使用
    if generate is not None:
        print("🚀 Fast-dLLMの正しいキャッシュなし実装を使用")
        outputs, nfe = generate(
            model, prompt, steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=temperature, remasking=remasking, mask_id=mask_id, threshold=threshold
        )

        # RoPEスケーリングを元に戻す
        if original_theta is not None:
            model.config.rope_theta = original_theta

        total_time = time.time() - start_time
        metrics = {
            'total_time': total_time,
            'tokens_per_second': gen_length / total_time if total_time > 0 else 0,
            'nfe': nfe,
            'cache_hits': 0,
            'total_tokens_generated': gen_length,
            'blocks_processed': gen_length // block_length,
            'scaling_factor_used': scaling_factor
        }
        return outputs, nfe, metrics

    # フォールバック: 独自実装（キャッシュなし）
    print("⚠️  Fast-dLLM実装利用不可、独自キャッシュなし実装を使用")

    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    total_tokens_generated = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block)

        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, current_block_end:] = 0

            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index, x,
                num_transfer_tokens[:,
                                    i] if threshold is None else None, threshold
            )
            x[transfer_index] = x0[transfer_index]
            total_tokens_generated += transfer_index.sum().item()
            i += 1

            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break

    # RoPEスケーリングを元に戻す
    if original_theta is not None:
        model.config.rope_theta = original_theta

    total_time = time.time() - start_time

    metrics = {
        'total_time': total_time,
        'tokens_per_second': total_tokens_generated / total_time if total_time > 0 else 0,
        'nfe': nfe,
        'cache_hits': 0,
        'total_tokens_generated': total_tokens_generated,
        'blocks_processed': num_blocks,
        'scaling_factor_used': scaling_factor
    }

    return x, nfe, metrics


def load_model_with_scaling(model_path, scaling_factor=1, device='auto'):
    """
    RoPEスケーリング付きでLLaDAモデルを読み込み
    """
    print(f"📥 モデル読み込み: {model_path}")
    print(f"📏 RoPEスケーリング係数: {scaling_factor}")

    # 設定読み込み
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # RoPEスケーリング適用
    if scaling_factor > 1:
        original_theta = getattr(config, 'rope_theta', 10000.0)
        config.rope_theta = original_theta * scaling_factor
        print(f"🔧 RoPE θ: {original_theta} → {config.rope_theta}")

    # LLaDAモデル読み込み（Fast-dLLMの正しいクラス使用 または フォールバック）
    model = LLaDAModelLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )

    # トークナイザ読み込み
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)

    model.eval()
    return model, tokenizer, config


def format_metrics(metrics):
    """メトリクス整形表示"""
    print("\n📊 生成メトリクス:")
    print(f"  ⏱️  総時間: {metrics['total_time']:.2f}秒")
    print(f"  🚀 速度: {metrics['tokens_per_second']:.1f} tok/s")
    print(f"  🔄 NFE: {metrics['nfe']}")
    print(f"  💾 キャッシュヒット: {metrics['cache_hits']}")
    print(f"  📝 生成トークン数: {metrics['total_tokens_generated']}")
    print(f"  🧱 処理ブロック数: {metrics['blocks_processed']}")
    print(f"  📏 スケーリング係数: {metrics['scaling_factor_used']}")
