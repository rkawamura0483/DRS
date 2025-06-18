# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
import time

from .adaptive_scheduler import AdaptiveInferenceScheduler
from .cache_manager import TieredCacheManager, CacheTier
from .generate import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_transfer_index_with_confidence
)


@torch.no_grad()
def generate_with_adaptive_scheduling(
    model,
    prompt: torch.Tensor,
    gen_length: int = 128,
    base_block_size: int = 16,
    base_confidence_threshold: float = 0.8,
    adaptation_rate: float = 0.2,
    enable_tiered_cache: bool = True,
    temperature: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: Optional[int] = None,
    scheduler_config: Optional[Dict] = None,
    cache_config: Optional[Dict] = None,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Self-Correcting Adaptive Inference Scheduling for Diffusion LLMs

    アダプティブスケジューリングを使用した生成関数。動的ブロックサイズ調整、
    適応的信頼度閾値、階層キャッシュ管理を統合。

    Args:
        model: LLaDAモデル
        prompt: 入力プロンプト (1, prompt_length)
        gen_length: 生成する長さ
        base_block_size: 初期ブロックサイズ
        base_confidence_threshold: 初期信頼度閾値
        adaptation_rate: 適応率
        enable_tiered_cache: 階層キャッシュを有効にするか
        temperature: サンプリング温度
        remasking: リマスキング戦略
        mask_id: マスクトークンID
        scheduler_config: スケジューラー設定
        cache_config: キャッシュ設定
        verbose: 詳細出力

    Returns:
        (生成されたトークン, 詳細メトリクス)
    """

    # デフォルト設定
    scheduler_config = scheduler_config or {}
    cache_config = cache_config or {}

    if verbose:
        print(f"\n🚀 Adaptive Scheduling 開始")
        print(f"   プロンプト長: {prompt.shape[1]}")
        print(f"   生成長: {gen_length}")
        print(f"   初期ブロックサイズ: {base_block_size}")
        print(f"   初期信頼度閾値: {base_confidence_threshold}")
        print(f"   階層キャッシュ: {enable_tiered_cache}")

    # マスクIDの取得
    if mask_id is None:
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'mask_token_id'):
            mask_id = model.tokenizer.mask_token_id
        else:
            mask_id = model.config.mask_token_id

    # スケジューラーの初期化
    scheduler = AdaptiveInferenceScheduler(
        min_block_size=max(4, base_block_size // 4),
        max_block_size=min(64, base_block_size * 4),
        base_confidence_threshold=base_confidence_threshold,
        adaptation_sensitivity=adaptation_rate,
        **scheduler_config
    )

    # キャッシュマネージャーの初期化
    cache_manager = None
    if enable_tiered_cache:
        cache_manager = TieredCacheManager(**cache_config)

    # 初期設定
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # 初期ブロックサイズで開始
    current_block_size = base_block_size
    current_threshold = base_confidence_threshold

    # メトリクス収集
    metrics = {
        'nfe': 0,
        'total_adaptations': 0,
        'block_size_history': [],
        'threshold_history': [],
        'confidence_history': [],
        'entropy_history': [],
        'cache_efficiency': {},
        'timing': {
            'total_time': 0,
            'adaptation_time': 0,
            'generation_time': 0,
            'cache_time': 0
        },
        'blocks_processed': 0,
        'tier_usage': {'tier1': 0, 'tier2': 0, 'tier3': 0}
    }

    start_time = time.time()

    # Phase 1: プロンプトキャッシュの設定
    if enable_tiered_cache:
        cache_start = time.time()
        output = model(x[:, :prompt.shape[1]], use_cache=True)
        cache_manager.set_prompt_cache(prompt.shape[1], output.past_key_values)
        metrics['timing']['cache_time'] += time.time() - cache_start
        metrics['nfe'] += 1

        if verbose:
            print(f"✅ Tier1キャッシュ設定完了 (プロンプト長: {prompt.shape[1]})")

    # Phase 2: アダプティブ生成
    generated_tokens = 0
    block_id = 0

    with tqdm(total=gen_length, desc="Adaptive Generation", disable=not verbose) as pbar:
        while generated_tokens < gen_length:
            block_start_time = time.time()

            # 現在のブロック範囲を決定
            block_start = prompt.shape[1] + generated_tokens
            remaining_tokens = gen_length - generated_tokens
            actual_block_size = min(current_block_size, remaining_tokens)
            block_end = block_start + actual_block_size

            if verbose and block_id % 5 == 0:
                print(
                    f"\n📦 ブロック {block_id}: サイズ={actual_block_size}, 閾値={current_threshold:.3f}")

            # ブロック生成
            block_generated, block_metrics = _generate_block_adaptive(
                model=model,
                x=x,
                block_start=block_start,
                block_end=block_end,
                current_threshold=current_threshold,
                temperature=temperature,
                remasking=remasking,
                cache_manager=cache_manager,
                block_id=block_id,
                mask_id=mask_id
            )

            metrics['nfe'] += block_metrics['nfe']
            metrics['timing']['generation_time'] += block_metrics['generation_time']

            # アダプティブ調整
            if block_metrics['confidence_scores'] is not None:
                adaptation_start = time.time()

                # スケジューラーによる適応
                next_block_size, adapted_threshold, step_metrics = scheduler.step(
                    logits=block_metrics['final_logits'],
                    tokens=x[:, block_start:block_end],
                    mask_index=block_metrics['final_mask_index'],
                    step_num=block_id,
                    total_steps=gen_length // base_block_size
                )

                # 適応検出
                if (next_block_size != current_block_size or
                        abs(adapted_threshold - current_threshold) > 0.01):
                    metrics['total_adaptations'] += 1
                    if verbose:
                        print(f"🔄 適応: ブロックサイズ {current_block_size}→{next_block_size}, "
                              f"閾値 {current_threshold:.3f}→{adapted_threshold:.3f}")

                # 更新
                current_block_size = next_block_size
                current_threshold = adapted_threshold

                # メトリクス記録
                metrics['block_size_history'].append(actual_block_size)
                metrics['threshold_history'].append(current_threshold)
                metrics['confidence_history'].append(
                    step_metrics['confidence'])
                metrics['entropy_history'].append(step_metrics['entropy'])

                metrics['timing']['adaptation_time'] += time.time() - \
                    adaptation_start

            # キャッシュ使用状況の記録
            if enable_tiered_cache and block_metrics['cache_tier']:
                if block_metrics['cache_tier'] == CacheTier.FROZEN:
                    metrics['tier_usage']['tier1'] += 1
                elif block_metrics['cache_tier'] == CacheTier.STABLE:
                    metrics['tier_usage']['tier2'] += 1
                elif block_metrics['cache_tier'] == CacheTier.ACTIVE:
                    metrics['tier_usage']['tier3'] += 1

            # 進捗更新
            generated_tokens += actual_block_size
            metrics['blocks_processed'] += 1
            block_id += 1

            pbar.update(actual_block_size)
            pbar.set_postfix({
                'B': actual_block_size,
                'τ': f"{current_threshold:.2f}",
                'NFE': metrics['nfe'],
                'Adapt': metrics['total_adaptations']
            })

            metrics['timing']['generation_time'] += time.time() - \
                block_start_time

    # 最終メトリクス計算
    metrics['timing']['total_time'] = time.time() - start_time

    # スケジューラーメトリクス
    scheduler_metrics = scheduler.get_adaptation_metrics()
    metrics.update({
        'avg_block_size': scheduler_metrics['current_block_size'],
        'final_threshold': scheduler_metrics['current_threshold'],
        'adaptation_rate': scheduler_metrics['adaptation_count'] / max(1, scheduler_metrics['total_blocks'])
    })

    # キャッシュメトリクス
    if enable_tiered_cache:
        cache_metrics = cache_manager.get_cache_efficiency_metrics()
        metrics['cache_efficiency'] = cache_metrics

        if verbose:
            cache_manager.print_cache_status()

    # 最終レポート
    if verbose:
        print(f"\n🎉 Adaptive Scheduling 完了!")
        print(f"   総時間: {metrics['timing']['total_time']:.2f}秒")
        print(f"   総NFE: {metrics['nfe']}")
        print(f"   適応回数: {metrics['total_adaptations']}")
        print(f"   平均ブロックサイズ: {np.mean(metrics['block_size_history']):.1f}")
        print(f"   最終信頼度閾値: {metrics['final_threshold']:.3f}")
        if enable_tiered_cache:
            print(f"   キャッシュヒット率: {cache_metrics['cache_hit_rate']:.2%}")

    return x, metrics


def _generate_block_adaptive(
    model,
    x: torch.Tensor,
    block_start: int,
    block_end: int,
    current_threshold: float,
    temperature: float,
    remasking: str,
    cache_manager: Optional[TieredCacheManager],
    block_id: int,
    mask_id: int
) -> Tuple[bool, Dict[str, Any]]:
    """
    アダプティブスケジューリング用ブロック生成

    Args:
        model: LLaDAモデル
        x: 現在のトークンテンソル
        block_start: ブロック開始位置
        block_end: ブロック終了位置
        current_threshold: 現在の信頼度閾値
        temperature: サンプリング温度
        remasking: リマスキング戦略
        cache_manager: キャッシュマネージャー
        block_id: ブロックID
        mask_id: マスクトークンID

    Returns:
        (生成成功フラグ, ブロックメトリクス)
    """
    block_start_time = time.time()
    nfe = 0
    confidence_scores = None
    final_logits = None
    final_mask_index = None
    cache_tier = None

    # ブロック内のマスクインデックス
    block_mask_index = (x[:, block_start:block_end] == mask_id)
    if not block_mask_index.any():
        # すでに生成済み
        return True, {
            'nfe': 0,
            'generation_time': 0,
            'confidence_scores': None,
            'final_logits': None,
            'final_mask_index': None,
            'cache_tier': None
        }

    # キャッシュの取得
    past_key_values = None
    if cache_manager is not None:
        past_key_values = cache_manager.get_cache_for_block(block_id)
        if past_key_values is None:
            past_key_values = cache_manager.get_base_cache()

    # 初期推論
    if past_key_values is not None:
        # キャッシュありの場合は部分的に推論
        output = model(x[:, block_start:block_end],
                       past_key_values=past_key_values, use_cache=True)
    else:
        # キャッシュなしの場合は全体を推論
        output = model(x, use_cache=True)

    nfe += 1
    final_logits = output.logits

    # マスク位置の更新
    mask_index = (x == mask_id)
    mask_index[:, block_end:] = 0  # ブロック範囲外はマスクしない
    final_mask_index = mask_index

    # トークン生成（信頼度付き）
    x0, transfer_index, confidence_probs = get_transfer_index_with_confidence(
        final_logits, temperature, remasking, mask_index, x,
        mask_index.sum(dim=1, keepdim=True)
    )

    # 信頼度フィルタリング
    if confidence_probs is not None:
        confidence_mask = confidence_probs >= current_threshold
        final_transfer_index = transfer_index & confidence_mask.unsqueeze(0)
    else:
        final_transfer_index = transfer_index

    # トークン更新
    x[final_transfer_index] = x0[final_transfer_index]

    # 信頼度スコアの計算
    if confidence_probs is not None:
        block_confidences = confidence_probs[:, block_start:block_end]
        confidence_scores = block_confidences[block_mask_index[0]]

    # キャッシュ更新
    if cache_manager is not None and confidence_scores is not None:
        cache_tier = cache_manager.update_cache(
            block_id=block_id,
            start_pos=block_start,
            end_pos=block_end,
            past_key_values=output.past_key_values,
            confidence_scores=confidence_scores
        )

    return True, {
        'nfe': nfe,
        'generation_time': time.time() - block_start_time,
        'confidence_scores': confidence_scores,
        'final_logits': final_logits,
        'final_mask_index': final_mask_index,
        'cache_tier': cache_tier
    }


@torch.no_grad()
def generate_with_custom_scheduler(
    model,
    prompt: torch.Tensor,
    scheduler: AdaptiveInferenceScheduler,
    cache_manager: Optional[TieredCacheManager] = None,
    gen_length: int = 128,
    temperature: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: Optional[int] = None,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    カスタムスケジューラーを使用した生成関数

    Args:
        model: LLaDAモデル
        prompt: 入力プロンプト
        scheduler: カスタムアダプティブスケジューラー
        cache_manager: カスタムキャッシュマネージャー
        gen_length: 生成長
        temperature: サンプリング温度
        remasking: リマスキング戦略
        mask_id: マスクトークンID
        verbose: 詳細出力

    Returns:
        (生成されたトークン, 詳細メトリクス)
    """
    return generate_with_adaptive_scheduling(
        model=model,
        prompt=prompt,
        gen_length=gen_length,
        base_block_size=scheduler.current_block_size,
        base_confidence_threshold=scheduler.current_threshold,
        enable_tiered_cache=cache_manager is not None,
        temperature=temperature,
        remasking=remasking,
        mask_id=mask_id,
        verbose=verbose
    )
