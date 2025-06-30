"""
Fast-dLLM × LongLLaDA 統合生成機能
Fast-dLLMの正しいデュアルキャッシュとLongLLaDAのRoPEスケーリングを組み合わせた高速長文生成
"""

from model.modeling_llada import LLaDAModelLM
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple, List
import time
import sys

# Fast-dLLMのインポート
sys.path.append('Fast-dLLM/llada')


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

    # 初期化
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    total_tokens_generated = 0
    cache_hits = 0

    print(
        f"📦 ブロック数: {num_blocks}, ブロック長: {block_length}, ブロック毎ステップ: {steps_per_block}")

    # ブロック単位生成（Fast-dLLMのデュアルキャッシュ実装）
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        print(f"🔄 ブロック {num_block + 1}/{num_blocks} 処理中...")

        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block)

        # Fast-dLLMのデュアルキャッシュ初期化
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        x0, transfer_index = get_transfer_index(
            output.logits, temperature, remasking, mask_index, x,
            num_transfer_tokens[:, 0] if threshold is None else None, threshold
        )
        x[transfer_index] = x0[transfer_index]
        nfe += 1
        total_tokens_generated += transfer_index.sum().item()

        # ブロック内反復（デュアルキャッシュ使用）
        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1

        while True:
            nfe += 1
            mask_index = (
                x[:, current_block_start:current_block_end] == mask_id)

            if mask_index.sum() == 0:
                break

            # デュアルキャッシュを使用した部分生成
            logits = model(
                x[:, current_block_start:current_block_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position
            ).logits
            cache_hits += 1

            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index,
                x[:, current_block_start:current_block_end],
                num_transfer_tokens[:, min(
                    i, steps_per_block-1)] if threshold is None else None,
                threshold
            )
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            total_tokens_generated += transfer_index.sum().item()
            i += 1

            if i >= steps_per_block:
                break

    # RoPEスケーリングを元に戻す
    if original_theta is not None:
        model.config.rope_theta = original_theta

    total_time = time.time() - start_time

    # メトリクス計算
    metrics = {
        'total_time': total_time,
        'tokens_per_second': total_tokens_generated / total_time if total_time > 0 else 0,
        'nfe': nfe,
        'cache_hits': cache_hits,
        'total_tokens_generated': total_tokens_generated,
        'blocks_processed': num_blocks,
        'scaling_factor_used': scaling_factor
    }

    return x, nfe, metrics


@torch.no_grad()
def generate_fast_long_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=32,
                                    temperature=0., remasking='low_confidence', mask_id=126336,
                                    threshold=None, scaling_factor=1):
    """
    Fast-dLLM × LongLLaDA 統合生成関数（プレフィックスキャッシュ使用）
    """
    start_time = time.time()
    device = model.device

    # RoPEスケーリング適用
    original_theta = None
    if scaling_factor > 1 and hasattr(model.config, 'rope_theta'):
        original_theta = model.config.rope_theta
        model.config.rope_theta = original_theta * scaling_factor
        print(f"🔧 RoPE θ スケーリング: {original_theta} → {model.config.rope_theta}")

    # 初期化
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0
    total_tokens_generated = 0
    cache_hits = 0

    print(
        f"📦 ブロック数: {num_blocks}, ブロック長: {block_length}, ブロック毎ステップ: {steps_per_block}")

    # ブロック単位生成（Fast-dLLMのプレフィックスキャッシュ実装）
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        print(f"🔄 ブロック {num_block + 1}/{num_blocks} 処理中...")

        block_mask_index = (
            x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block)

        # プレフィックスキャッシュ初期化
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        x0, transfer_index = get_transfer_index(
            output.logits, temperature, remasking, mask_index, x,
            num_transfer_tokens[:, 0] if threshold is None else None, threshold
        )
        x[transfer_index] = x0[transfer_index]
        total_tokens_generated += transfer_index.sum().item()

        # プレフィックス部分のキャッシュのみ保持
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i]
                                           [j][:, :, :current_block_start],)

        past_key_values = new_past_key_values
        nfe += 1

        # ブロック内反復
        i = 1
        while True:
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            if mask_index.sum() == 0:
                break

            logits = model(
                x[:, current_block_start:],
                past_key_values=past_key_values,
                use_cache=True
            ).logits
            cache_hits += 1

            x0, transfer_index = get_transfer_index(
                logits, temperature, remasking, mask_index,
                x[:, current_block_start:],
                num_transfer_tokens[:, min(
                    i, steps_per_block-1)] if threshold is None else None,
                threshold
            )
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            total_tokens_generated += transfer_index.sum().item()

            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            i += 1

    # RoPEスケーリングを元に戻す
    if original_theta is not None:
        model.config.rope_theta = original_theta

    total_time = time.time() - start_time

    # メトリクス計算
    metrics = {
        'total_time': total_time,
        'tokens_per_second': total_tokens_generated / total_time if total_time > 0 else 0,
        'nfe': nfe,
        'cache_hits': cache_hits,
        'total_tokens_generated': total_tokens_generated,
        'blocks_processed': num_blocks,
        'scaling_factor_used': scaling_factor
    }

    return x, nfe, metrics


def generate_fast_long(model, prompt, steps=128, gen_length=128, block_length=32,
                       temperature=0., remasking='low_confidence', mask_id=126336,
                       threshold=None, use_cache=True, scaling_factor=1, dual_cache=True):
    """
    統合生成関数（キャッシュ方式を選択可能）
    """
    if use_cache:
        if dual_cache:
            return generate_fast_long_dual_cache(
                model, prompt, steps, gen_length, block_length,
                temperature, remasking, mask_id, threshold, scaling_factor
            )
        else:
            return generate_fast_long_prefix_cache(
                model, prompt, steps, gen_length, block_length,
                temperature, remasking, mask_id, threshold, scaling_factor
            )
    else:
        # キャッシュなし実装（標準のFast-dLLM generate関数）
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

    # LLaDAモデル読み込み（Fast-dLLMの正しいクラス使用）
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
