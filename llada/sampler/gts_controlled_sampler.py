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

"""
GTS制御サンプラー実装

Generative Trajectory Stability (GTS) に基づいて、
パラレル生成の品質を動的に制御するサンプリング手法です。

主な機能：
- parallel-k トークン生成
- 内部軌跡の記録とGTS計算  
- 低安定性トークンの検出と再マスク
- 適応的な反復制御
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
from tqdm import tqdm

# llada内部モジュールのimport - 相対インポートと絶対インポートの両方に対応
try:
    # 相対インポート（パッケージとして実行される場合）
    from ..metrics.gts import BasicGTS, create_gts_metric, evaluate_trajectory_stability
    from ..utils.trajectory_recorder import TrajectoryRecorder
    try:
        from ..cache_manager import TieredCacheManager
        CACHE_AVAILABLE = True
    except ImportError:
        CACHE_AVAILABLE = False
except (ImportError, ValueError):
    # 絶対インポート（直接実行される場合）
    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from metrics.gts import BasicGTS, create_gts_metric, evaluate_trajectory_stability
    from utils.trajectory_recorder import TrajectoryRecorder
    try:
        from cache_manager import TieredCacheManager
        CACHE_AVAILABLE = True
    except ImportError:
        CACHE_AVAILABLE = False


def add_gumbel_noise(logits, temperature):
    """
    Gumbel Maxサンプリング用のノイズ追加
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    各ステップで遷移すべきトークン数を事前計算
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps,
                                      device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_transfer_index_gts(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    """
    GTS制御用の遷移インデックス取得（confidence情報も返す）
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

    return x0, transfer_index, x0_p


@torch.no_grad()
def generate_with_gts_controlled_sampling(
    model,
    prompt: torch.Tensor,
    gen_length: int = 128,
    parallel_k: int = 16,  # 並列生成するトークン数
    gts_threshold: float = 0.7,  # GTS閾値（これ以下で再マスク）
    max_iterations: int = 5,  # 最大反復回数
    steps_per_iteration: int = 8,  # 各反復内のdiffusionステップ数
    gts_metric_type: str = 'basic',  # GTSメトリクスタイプ
    temperature: float = 0.0,
    remasking: str = 'low_confidence',
    mask_id: Optional[int] = None,
    enable_tiered_cache: bool = True,
    verbose: bool = True,
    convergence_window: int = 3  # 収束判定ウィンドウサイズ
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    GTS制御サンプリング：idea.md 3.2節の実装

    Algorithm:
    1. parallel-k個のトークンをparallel生成
    2. 内部denoising軌跡を記録
    3. P-GTSスコアを計算
    4. GTS < τ のトークンを特定してマスク
    5. マスクされたトークンがあれば再実行、なければ終了

    Args:
        model: LLaDAモデル
        prompt: 入力プロンプト
        gen_length: 生成長
        parallel_k: 並列生成トークン数
        gts_threshold: GTS安定性閾値
        max_iterations: 最大反復回数
        steps_per_iteration: 各反復のdiffusionステップ数
        gts_metric_type: GTSメトリクスタイプ
        temperature: サンプリング温度
        remasking: リマスキング戦略
        mask_id: マスクトークンID
        enable_tiered_cache: 階層キャッシュを使用するか
        verbose: 詳細ログを出力するか
        convergence_window: 収束判定ウィンドウサイズ

    Returns:
        (generated_tokens, metrics_dict)
    """

    # マスクIDの取得
    if mask_id is None:
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'mask_token_id'):
            mask_id = model.tokenizer.mask_token_id
        elif hasattr(model.config, 'mask_token_id'):
            mask_id = model.config.mask_token_id
        else:
            mask_id = 126336  # LLaDAのデフォルト

    if verbose:
        print(f"\n🎯 GTS制御サンプリング開始")
        print(f"   プロンプト長: {prompt.shape[1]}")
        print(f"   生成長: {gen_length}")
        print(f"   並列k: {parallel_k}")
        print(f"   GTS閾値: {gts_threshold}")
        print(f"   最大反復: {max_iterations}")
        print(f"   GTSタイプ: {gts_metric_type}")

    # 初期化
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    # メトリクス初期化
    metrics = {
        'nfe': 0,
        'total_iterations': 0,
        'gts_scores_history': [],
        'unstable_tokens_history': [],
        'remasked_tokens_history': [],
        'convergence_history': [],
        'timing': {'total_time': 0, 'gts_computation_time': 0},
        'block_metrics': []
    }

    # キャッシュマネージャー初期化
    cache_manager = None
    if enable_tiered_cache and CACHE_AVAILABLE:
        cache_manager = TieredCacheManager()

    start_time = time.time()

    # ブロック単位で処理
    assert gen_length % parallel_k == 0, f"gen_length ({gen_length}) must be divisible by parallel_k ({parallel_k})"
    num_blocks = gen_length // parallel_k

    for block_idx in range(num_blocks):
        block_start = prompt.shape[1] + block_idx * parallel_k
        block_end = block_start + parallel_k

        if verbose:
            print(
                f"\n📦 ブロック {block_idx+1}/{num_blocks} (位置 {block_start}-{block_end})")

        # ブロック用メトリクス
        block_metrics = {
            'block_idx': block_idx,
            'iterations': 0,
            'final_gts_score': 0.0,
            'total_remasked': 0,
            'converged': False
        }

        # ブロック生成の反復処理
        for iteration in range(max_iterations):
            if verbose:
                print(f"   🔄 反復 {iteration+1}/{max_iterations}")

            # 軌跡記録器の初期化
            trajectory_recorder = TrajectoryRecorder(
                max_steps=steps_per_iteration + 5,
                record_mode='logits',  # GTSには logits が必要
                device=model.device
            )

            # 現在ブロックのマスク状況確認
            block_mask_index = (x[:, block_start:block_end] == mask_id)
            if block_mask_index.sum() == 0:
                if verbose:
                    print(f"   ✅ ブロック完了 (マスクトークンなし)")
                break

            # diffusion デノイジング実行（軌跡記録付き）
            iteration_nfe = 0
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_iteration)

            step_idx = 0
            while True:
                iteration_nfe += 1
                metrics['nfe'] += 1

                # モデル推論
                mask_index = (x == mask_id)
                mask_index[:, block_end:] = 0  # 現在ブロック以降をマスク

                logits = model(x).logits

                # 軌跡記録（現在ブロックのみ）
                block_logits = logits[:, block_start:block_end, :]
                block_mask = mask_index[:, block_start:block_end]
                trajectory_recorder.record_step(
                    block_logits,
                    step_idx,
                    block_mask,
                    metadata={'iteration': iteration, 'nfe': iteration_nfe}
                )

                # トークン更新
                x0, transfer_index, confidence = get_transfer_index_gts(
                    logits, temperature, remasking, mask_index, x,
                    num_transfer_tokens[:, step_idx] if step_idx < num_transfer_tokens.size(
                        1) else torch.tensor([1]),
                    threshold=None
                )
                x[transfer_index] = x0[transfer_index]

                # ブロック完了チェック
                if (x[:, block_start:block_end] == mask_id).sum() == 0:
                    break

                step_idx += 1
                if step_idx >= steps_per_iteration:
                    break

            # GTS計算
            gts_start_time = time.time()

            # 軌跡データからGTSメトリクス計算
            logits_sequence = trajectory_recorder.get_logits_sequence_for_gts()

            if verbose:
                print(
                    f"   📊 軌跡データ: {len(logits_sequence)}ステップ, NFE={iteration_nfe}")

            if len(logits_sequence) >= 2:
                try:
                    # GTSメトリクスを使用して安定性を評価
                    gts_metric = create_gts_metric(gts_metric_type)

                    # 各ステップのlogitsを渡してGTS更新
                    for step, step_logits in enumerate(logits_sequence):
                        # step_logits shape: (1, parallel_k, vocab_size)
                        # マスクされたトークンのみを評価
                        current_mask = (
                            x[:, block_start:block_end] == mask_id) if iteration == 0 else None
                        gts_metric.update(step_logits, step, current_mask)

                    gts_score = gts_metric.value()

                    if verbose:
                        print(
                            f"   📊 GTS詳細: ステップ数={gts_metric.num_steps}, トークン数={gts_metric.num_tokens}")
                        if hasattr(gts_metric, 'total_flips'):
                            print(f"   📊 フリップ数: {gts_metric.total_flips}")

                except Exception as e:
                    if verbose:
                        print(f"   ⚠️ GTS計算エラー: {e}")
                    gts_score = 1.0  # エラー時は安定と仮定

            else:
                if verbose:
                    print(f"   ⚠️ 軌跡データ不足: {len(logits_sequence)}ステップ")
                gts_score = 1.0  # データ不足時は安定と仮定

            gts_computation_time = time.time() - gts_start_time
            metrics['timing']['gts_computation_time'] += gts_computation_time

            # GTS結果の記録
            metrics['gts_scores_history'].append(gts_score)
            block_metrics['final_gts_score'] = gts_score

            if verbose:
                print(f"   📊 GTS スコア: {gts_score:.4f} (閾値: {gts_threshold})")

            # 安定性チェック
            if gts_score >= gts_threshold:
                if verbose:
                    print(f"   ✅ 安定性達成 (GTS >= {gts_threshold})")
                block_metrics['converged'] = True
                break

            # 不安定な場合の処理
            if iteration < max_iterations - 1:  # 最後の反復でない場合
                # 軌跡分析から不安定トークンを特定
                stability_data = trajectory_recorder.get_stability_data()

                if stability_data['has_sufficient_data']:
                    # 最も不安定なトークンを特定
                    unstable_tokens = []
                    for token_idx, instability in stability_data['most_unstable_tokens']:
                        if instability > 0:  # 何らかの変更があったトークン
                            actual_pos = block_start + token_idx
                            if actual_pos < block_end:
                                unstable_tokens.append(actual_pos)

                    # 不安定トークンを再マスク
                    if unstable_tokens:
                        for pos in unstable_tokens[:parallel_k//2]:  # 最大半分を再マスク
                            x[0, pos] = mask_id

                        remasked_count = len(unstable_tokens[:parallel_k//2])
                        block_metrics['total_remasked'] += remasked_count
                        metrics['remasked_tokens_history'].append(
                            remasked_count)

                        if verbose:
                            print(f"   🔄 {remasked_count}個のトークンを再マスク")
                    else:
                        if verbose:
                            print(f"   ⚠️ 不安定トークンが特定できず、反復終了")
                        break
                else:
                    if verbose:
                        print(f"   ⚠️ 軌跡データ不足、反復終了")
                    break

            block_metrics['iterations'] = iteration + 1

        # ブロック処理完了
        metrics['total_iterations'] += block_metrics['iterations']
        metrics['block_metrics'].append(block_metrics)

        if verbose:
            print(f"   🏁 ブロック {block_idx+1} 完了: "
                  f"反復={block_metrics['iterations']}, "
                  f"GTS={block_metrics['final_gts_score']:.4f}")

    # 全体完了
    metrics['timing']['total_time'] = time.time() - start_time

    # 最終統計
    all_gts_scores = metrics['gts_scores_history']
    metrics['final_stats'] = {
        'avg_gts_score': np.mean(all_gts_scores) if all_gts_scores else 0.0,
        'min_gts_score': np.min(all_gts_scores) if all_gts_scores else 0.0,
        'max_gts_score': np.max(all_gts_scores) if all_gts_scores else 0.0,
        'total_remasked': sum(metrics['remasked_tokens_history']),
        'avg_iterations_per_block': metrics['total_iterations'] / num_blocks if num_blocks > 0 else 0,
        'blocks_converged': sum(1 for b in metrics['block_metrics'] if b['converged'])
    }

    if verbose:
        print(f"\n🎉 GTS制御サンプリング完了")
        print(f"   総NFE: {metrics['nfe']}")
        print(f"   総反復: {metrics['total_iterations']}")
        print(f"   平均GTS: {metrics['final_stats']['avg_gts_score']:.4f}")
        print(
            f"   収束ブロック: {metrics['final_stats']['blocks_converged']}/{num_blocks}")
        print(f"   実行時間: {metrics['timing']['total_time']:.2f}秒")

    return x, metrics


@torch.no_grad()
def generate_with_gts_simple(
    model,
    prompt: torch.Tensor,
    gen_length: int = 32,
    gts_threshold: float = 0.8,
    max_iterations: int = 3,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    シンプルなGTS制御生成（デバッグ用）

    全シーケンスを一度に処理し、不安定部分のみ再実行します。
    """

    # マスクIDの取得
    mask_id = 126336  # LLaDAデフォルト

    # 初期化
    x = torch.full((1, prompt.shape[1] + gen_length),
                   mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    metrics = {'nfe': 0, 'iterations': 0, 'gts_scores': []}

    if verbose:
        print(f"🎯 シンプルGTS制御生成開始 (長さ: {gen_length})")

    for iteration in range(max_iterations):
        metrics['iterations'] += 1

        # 軌跡記録
        recorder = TrajectoryRecorder(record_mode='logits')

        # 単純な反復デノイジング
        steps = 5
        for step in range(steps):
            metrics['nfe'] += 1

            mask_index = (x == mask_id)
            if mask_index.sum() == 0:
                break

            logits = model(x).logits

            # 軌跡記録
            recorder.record_step(logits, step, mask_index)

            # サンプリング
            logits_with_noise = add_gumbel_noise(logits, temperature=0.0)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # 低信頼度トークンを更新
            x = torch.where(mask_index, x0, x)

        # GTS計算
        logits_seq = recorder.get_logits_sequence_for_gts()
        if len(logits_seq) >= 2:
            basic_gts = BasicGTS()
            for step, step_logits in enumerate(logits_seq):
                basic_gts.update(step_logits, step)
            gts_score = basic_gts.value()
        else:
            gts_score = 1.0

        metrics['gts_scores'].append(gts_score)

        if verbose:
            print(f"   反復 {iteration+1}: GTS = {gts_score:.4f}")

        # 安定性チェック
        if gts_score >= gts_threshold:
            if verbose:
                print(f"   ✅ 安定性達成!")
            break

        # 最低信頼度の部分を再マスク（最後の反復以外）
        if iteration < max_iterations - 1:
            # シンプルな再マスク戦略
            remaining_masks = (x == mask_id).sum()
            if remaining_masks > 0:
                continue  # まだマスクがあるので続行
            else:
                # 全て埋まった場合、不安定な位置を再マスク
                stability_data = recorder.get_stability_data()
                if stability_data['has_sufficient_data']:
                    unstable_positions = []
                    for token_idx, instability in stability_data['most_unstable_tokens'][:gen_length//4]:
                        if instability > 0:
                            pos = prompt.shape[1] + token_idx
                            if pos < x.shape[1]:
                                unstable_positions.append(pos)

                    for pos in unstable_positions:
                        x[0, pos] = mask_id

                    if verbose and unstable_positions:
                        print(f"   🔄 {len(unstable_positions)}個を再マスク")

    if verbose:
        print(
            f"🏁 完了: {metrics['iterations']}反復, 最終GTS = {metrics['gts_scores'][-1]:.4f}")

    return x, metrics
