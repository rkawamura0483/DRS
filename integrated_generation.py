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
import subprocess

# Fast-dLLMのパスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"🔍 現在のディレクトリ: {current_dir}")


def auto_clone_repositories():
    """Fast-dLLMとLongLLaDAリポジトリを自動クローン"""
    repositories = [
        {
            "name": "Fast-dLLM",
            "url": "https://github.com/NVlabs/Fast-dLLM.git",
            "dir": os.path.join(current_dir, "Fast-dLLM")
        },
        {
            "name": "LongLLaDA",
            "url": "https://github.com/OpenMOSS/LongLLaDA.git",
            "dir": os.path.join(current_dir, "LongLLaDA")
        }
    ]

    cloned_any = False
    for repo in repositories:
        if not os.path.exists(repo['dir']):
            print(f"📥 {repo['name']} が見つかりません。クローン中...")
            try:
                subprocess.run(
                    f"git clone {repo['url']} {repo['dir']}",
                    shell=True, check=True, cwd=current_dir
                )
                print(f"✅ {repo['name']} のクローン完了")
                cloned_any = True
            except subprocess.CalledProcessError as e:
                print(f"❌ {repo['name']} のクローン失敗: {e}")

    return cloned_any


# 必要なリポジトリが存在しない場合は自動クローン
if not os.path.exists(os.path.join(current_dir, 'Fast-dLLM')) or not os.path.exists(os.path.join(current_dir, 'LongLLaDA')):
    print("🔄 必要なリポジトリが見つかりません。自動セットアップを開始...")
    auto_clone_repositories()

# modeling_llada.pyファイルを探す
print("\n🔍 modeling_llada.py を検索中...")
model_path = None
generate_path = None

for root, dirs, files in os.walk(current_dir):
    if 'modeling_llada.py' in files:
        print(f"✅ 見つかりました: {root}")
        model_path = root
        generate_path = os.path.dirname(
            root) if root.endswith('model') else root
        break
else:
    # フォールバック: 標準的なパス
    print("🔍 標準パスを使用...")
    model_path = os.path.join(current_dir, 'Fast-dLLM', 'llada', 'model')
    generate_path = os.path.join(current_dir, 'Fast-dLLM', 'llada')

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
    転送インデックスを取得（メモリ最適化版）

    Args:
        logits: モデルのlogits出力 (B, L, V)
        temperature: サンプリング温度
        remasking: リマスキング戦略
        mask_index: マスクインデックス
        x: 現在の入力
        num_transfer_tokens: 転送トークン数
        threshold: 信頼度閾値
    """
    # メモリ最適化: float16を維持し、必要な部分のみ処理
    device = logits.device
    dtype = logits.dtype

    # Gumbelノイズを追加してargmaxを取得
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == 'low_confidence':
        # メモリ最適化: チャンク処理でsoftmaxを計算
        chunk_size = 512  # チャンクサイズを調整可能
        p_values = []

        # マスクされた位置のみを処理
        masked_positions = mask_index.any(dim=0)

        for i in range(0, logits.shape[1], chunk_size):
            end_idx = min(i + chunk_size, logits.shape[1])

            # このチャンクにマスクされた位置があるかチェック
            if not masked_positions[i:end_idx].any():
                # マスクされた位置がない場合はスキップ
                p_values.append(torch.zeros((logits.shape[0], end_idx - i),
                                            device=device, dtype=dtype))
                continue

            # チャンクのlogitsを処理（float32で計算してfloat16に戻す）
            chunk_logits = logits[:, i:end_idx].float()
            chunk_p = F.softmax(chunk_logits, dim=-1)

            # 対応するx0の値を取得
            chunk_x0 = x0[:, i:end_idx].unsqueeze(-1)
            chunk_x0_p = torch.gather(
                chunk_p, dim=-1, index=chunk_x0).squeeze(-1)

            p_values.append(chunk_x0_p.to(dtype))

            # メモリ解放
            del chunk_logits, chunk_p, chunk_x0_p

        x0_p = torch.cat(p_values, dim=1)

    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]),
                          device=device, dtype=dtype)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -float('inf'))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)

    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    # バッチ処理の最適化
    for j in range(confidence.shape[0]):
        valid_confidence = confidence[j][confidence[j] != -float('inf')]
        if len(valid_confidence) == 0:
            continue

        num_tokens = num_transfer_tokens[j].item(
        ) if num_transfer_tokens is not None else len(valid_confidence)
        num_tokens = min(num_tokens, len(valid_confidence))

        if num_tokens > 0:
            _, select_index = torch.topk(
                confidence[j], k=num_tokens, largest=True)
            transfer_index[j, select_index] = True

            if threshold is not None:
                # 閾値以下の信頼度を持つトークンを除外
                low_conf_mask = confidence[j, select_index] < threshold
                transfer_index[j, select_index[low_conf_mask]] = False

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
    メモリ最適化版
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

    # メモリ最適化: 長文の場合はblock_lengthを動的調整
    input_length = prompt.shape[1]
    if input_length > 2000:  # 長文の場合
        block_length = max(64, block_length)  # より大きなブロックサイズ使用
        steps = min(64, steps)  # ステップ数を減らしてメモリ節約
        print(f"📏 長文対応: block_length={block_length}, steps={steps}")

    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # メモリ最適化: 必要に応じてgen_lengthを調整
    if gen_length % block_length != 0:
        gen_length = ((gen_length // block_length) + 1) * block_length
        print(f"📏 生成長を調整: {gen_length} (block_length={block_length}に合わせて)")

    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)

    nfe = 0
    total_tokens_generated = 0

    # メモリ監視と早期停止機能
    max_memory_mb = 35000  # 35GB制限（A100の場合）

    for num_block in range(num_blocks):
        # メモリ使用量チェック
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            if current_memory > max_memory_mb / 1024:
                print(f"⚠️  メモリ使用量が制限に近づいています: {current_memory:.1f}GB")
                torch.cuda.empty_cache()

        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block)

        i = 0
        max_iterations = steps_per_block * 2  # 無限ループ防止

        while i < max_iterations:
            nfe += 1
            mask_index = (x == mask_id)

            # メモリ最適化: logitsの取得時にメモリ管理
            try:
                with torch.cuda.device(model.device):
                    logits = model(x).logits
                    # 現在のブロック以降をマスク
                    mask_index[:, current_block_end:] = 0

                    x0, transfer_index = get_transfer_index(
                        logits, temperature, remasking, mask_index, x,
                        num_transfer_tokens[:,
                                            i] if threshold is None else None, threshold
                    )

                    # メモリ解放
                    del logits

                    x[transfer_index] = x0[transfer_index]
                    total_tokens_generated += transfer_index.sum().item()
                    i += 1

                    # ブロック完了チェック
                    if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                        break

            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ CUDA OOM エラー (ブロック {num_block}, イテレーション {i}): {e}")
                # 緊急メモリ解放
                torch.cuda.empty_cache()
                # より保守的なパラメータで再試行
                if block_length > 16:
                    block_length = block_length // 2
                    print(f"🔧 block_lengthを縮小: {block_length}")
                    break
                else:
                    raise e

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


def get_memory_usage():
    """現在のGPUメモリ使用量を取得"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        total = torch.cuda.get_device_properties(
            0).total_memory / (1024**3)  # GB
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': total - allocated
        }
    return None


def optimize_parameters_for_memory(input_length, available_memory_gb=35):
    """
    入力長とメモリ容量に基づいて最適なパラメータを計算

    Args:
        input_length: 入力トークン数
        available_memory_gb: 利用可能メモリ（GB）

    Returns:
        dict: 最適化されたパラメータ
    """
    # デフォルトパラメータ
    params = {
        'steps': 64,
        'gen_length': 128,
        'block_length': 32,
        'remasking': 'low_confidence'
    }

    # 入力長に基づく調整
    if input_length > 8000:
        # 超長文（8k+）
        params.update({
            'steps': 8,
            'gen_length': 64,
            'block_length': 128,
            'remasking': 'random'
        })
    elif input_length > 4000:
        # 長文（4k+）
        params.update({
            'steps': 16,
            'gen_length': 96,
            'block_length': 96,
            'remasking': 'random'
        })
    elif input_length > 2000:
        # 中長文（2k+）
        params.update({
            'steps': 32,
            'gen_length': 128,
            'block_length': 64,
            'remasking': 'random'
        })

    # メモリ容量に基づく調整
    if available_memory_gb < 20:  # 20GB未満の場合
        params['steps'] = min(params['steps'], 16)
        params['gen_length'] = min(params['gen_length'], 64)
        params['block_length'] = max(params['block_length'], 64)

    return params


def safe_generate_with_fallback(model, prompt, **kwargs):
    """
    メモリ不足時のフォールバック機能付き安全生成
    """
    # 入力長チェック
    input_length = prompt.shape[1]
    memory_info = get_memory_usage()

    print(f"📏 入力長: {input_length} トークン")
    if memory_info:
        print(
            f"💾 メモリ使用量: {memory_info['allocated']:.1f}GB / {memory_info['total']:.1f}GB")

    # パラメータ最適化
    available_memory = memory_info['total'] - \
        5 if memory_info else 35  # 5GB余裕を持たせる
    optimized_params = optimize_parameters_for_memory(
        input_length, available_memory)

    # kwargsとマージ（ユーザー指定を優先）
    final_params = {**optimized_params, **kwargs}

    print(f"🔧 最適化パラメータ: {final_params}")

    # 段階的フォールバック試行
    fallback_configs = [
        final_params,  # 最適化済み
        # ステップ半減
        {**final_params, 'steps': max(8, final_params['steps'] // 2)},
        {**final_params, 'steps': 8, 'gen_length': 32, 'block_length': 64},  # 最小構成
    ]

    for i, config in enumerate(fallback_configs):
        try:
            if i > 0:
                print(f"🔄 フォールバック試行 {i}: {config}")
                torch.cuda.empty_cache()

            return generate_fast_long_dual_cache(
                model=model,
                prompt=prompt,
                **config
            )

        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ OOM エラー (試行 {i+1}): {e}")
            if i == len(fallback_configs) - 1:
                print("🆘 全ての設定でメモリ不足。プロセスを終了します。")
                raise e
            continue

    # ここには到達しないはず
    raise RuntimeError("予期しないエラー: フォールバック処理に失敗")
