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
Generative Trajectory Stability (GTS) メトリクス実装

diffusion LLMの生成過程における軌跡の安定性を測定するメトリクス群。
パラレル生成されたトークンの一貫性を評価するために使用される。

主な実装：
- BasicGTS: argmaxの変更回数に基づく基本的な安定性指標
- SemanticGTS: 埋め込みベクトルのコサイン距離を考慮した意味的安定性
- ProbabilisticGTS: Jensen-Shannon Divergenceに基づく確率分布の安定性
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

try:
    from scipy.spatial.distance import jensenshannon
    SCIPY_AVAILABLE = True
except (ImportError, ValueError):  # ValueErrorも含めてscipy問題をキャッチ
    SCIPY_AVAILABLE = False


class BaseGTSMetric(ABC):
    """
    GTS メトリクスの基底クラス

    すべてのGTSメトリクスは以下のインターフェースを実装する必要があります：
    - update(): 新しいステップの情報を追加
    - value(): 現在のGTSスコアを計算
    - reset(): 状態をリセット
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        基底クラスの初期化

        Args:
            device: 計算に使用するデバイス（CPU/GPU）
        """
        self.device = device if device is not None else torch.device('cpu')
        self.trajectory_data = []  # ステップごとのデータを格納
        self.num_tokens = 0
        self.num_steps = 0

    @abstractmethod
    def update(self, logits: torch.Tensor, step: int, token_positions: Optional[torch.Tensor] = None) -> None:
        """
        新しいステップの情報をメトリクスに追加

        Args:
            logits: (batch_size, seq_len, vocab_size) の logits テンソル
            step: 現在のステップ番号
            token_positions: 評価対象のトークン位置 (optional)
        """
        pass

    @abstractmethod
    def value(self) -> float:
        """
        現在のGTSスコアを計算

        Returns:
            0.0-1.0 の範囲のGTSスコア（高いほど安定）
        """
        pass

    def reset(self) -> None:
        """メトリクスの状態をリセット"""
        self.trajectory_data.clear()
        self.num_tokens = 0
        self.num_steps = 0


class BasicGTS(BaseGTSMetric):
    """
    基本的なGTSメトリクス

    argmaxトークンの変更回数に基づいて安定性を測定します。
    計算式: GTS = 1 - (総フリップ数) / (トークン数 × ステップ数)
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        BasicGTSの初期化

        Args:
            device: 計算に使用するデバイス
        """
        super().__init__(device)
        self.previous_argmax = None
        self.total_flips = 0

    def update(self, logits: torch.Tensor, step: int, token_positions: Optional[torch.Tensor] = None) -> None:
        """
        新しいステップの情報を追加し、argmaxの変更を記録

        Args:
            logits: (batch_size, seq_len, vocab_size) の logits テンソル
            step: 現在のステップ番号
            token_positions: 評価対象のトークン位置マスク
        """
        # 指定された位置のトークンのみを評価
        if token_positions is not None:
            eval_logits = logits[token_positions]
        else:
            eval_logits = logits.view(-1, logits.size(-1))

        current_argmax = torch.argmax(eval_logits, dim=-1)

        if self.previous_argmax is not None:
            # 前ステップからの変更をカウント
            flips = (current_argmax != self.previous_argmax).sum().item()
            self.total_flips += flips

        self.previous_argmax = current_argmax.clone()
        self.num_tokens = current_argmax.numel()
        self.num_steps = step + 1

    def value(self) -> float:
        """
        基本GTSスコアを計算

        Returns:
            0.0-1.0 の範囲のスコア（1.0が最も安定）
        """
        if self.num_steps == 0 or self.num_tokens == 0:
            return 1.0  # データが全くない場合のみ最大安定性と仮定

        # ステップ数が1の場合、フリップは発生し得ないので安定とみなす
        if self.num_steps == 1:
            return 1.0

        max_possible_flips = self.num_tokens * (self.num_steps - 1)
        if max_possible_flips == 0:
            return 1.0

        stability = 1.0 - (self.total_flips / max_possible_flips)
        return max(0.0, stability)  # 0以下にならないようにクリップ


class SemanticGTS(BaseGTSMetric):
    """
    意味的GTSメトリクス

    トークンの埋め込みベクトル間のコサイン距離を考慮して
    変更の意味的重要度を重み付けします。
    """

    def __init__(self, embedding_matrix: torch.Tensor, device: Optional[torch.device] = None):
        """
        SemanticGTSの初期化

        Args:
            embedding_matrix: (vocab_size, embed_dim) のトークン埋め込み行列
            device: 計算に使用するデバイス
        """
        super().__init__(device)
        self.embedding_matrix = embedding_matrix.to(self.device)
        self.previous_argmax = None
        self.total_semantic_distance = 0.0
        self.total_changes = 0

    def update(self, logits: torch.Tensor, step: int, token_positions: Optional[torch.Tensor] = None) -> None:
        """
        新しいステップの情報を追加し、意味的距離を計算

        Args:
            logits: (batch_size, seq_len, vocab_size) の logits テンソル
            step: 現在のステップ番号
            token_positions: 評価対象のトークン位置マスク
        """
        # 指定された位置のトークンのみを評価
        if token_positions is not None:
            eval_logits = logits[token_positions]
        else:
            eval_logits = logits.view(-1, logits.size(-1))

        current_argmax = torch.argmax(eval_logits, dim=-1)

        if self.previous_argmax is not None:
            # 変更があったトークンを特定
            changed_mask = (current_argmax != self.previous_argmax)

            if changed_mask.any():
                # 変更前後のトークンの埋め込みを取得
                old_tokens = self.previous_argmax[changed_mask]
                new_tokens = current_argmax[changed_mask]

                old_embeddings = self.embedding_matrix[old_tokens]
                new_embeddings = self.embedding_matrix[new_tokens]

                # コサイン距離を計算（1 - コサイン類似度）
                cos_sim = F.cosine_similarity(
                    old_embeddings, new_embeddings, dim=-1)
                semantic_distances = 1.0 - cos_sim

                self.total_semantic_distance += semantic_distances.sum().item()
                self.total_changes += changed_mask.sum().item()

        self.previous_argmax = current_argmax.clone()
        self.num_tokens = current_argmax.numel()
        self.num_steps = step + 1

    def value(self) -> float:
        """
        意味的GTSスコアを計算

        Returns:
            0.0-1.0 の範囲のスコア（1.0が最も安定）
        """
        if self.num_steps <= 1 or self.num_tokens == 0:
            return 1.0

        if self.total_changes == 0:
            return 1.0  # 変更がない場合は完全に安定

        # 平均意味的距離を計算
        avg_semantic_distance = self.total_semantic_distance / self.total_changes

        # 意味的距離は0-2の範囲なので、0-1にスケール
        normalized_distance = avg_semantic_distance / 2.0
        stability = 1.0 - normalized_distance

        return max(0.0, min(1.0, stability))


class ProbabilisticGTS(BaseGTSMetric):
    """
    確率的GTS メトリクス

    Jensen-Shannon Divergence を使用して、連続するステップ間の
    確率分布の安定性を測定します。最も精密なGTS測定手法です。
    """

    def __init__(self, device: Optional[torch.device] = None, temperature: float = 1.0):
        """
        ProbabilisticGTSの初期化

        Args:
            device: 計算に使用するデバイス
            temperature: softmax温度パラメータ
        """
        super().__init__(device)

        if not SCIPY_AVAILABLE:
            raise ImportError(
                "ProbabilisticGTS requires scipy. Please install scipy>=1.10")

        self.temperature = temperature
        self.previous_probs = None
        self.total_jsd = 0.0
        self.total_comparisons = 0

    def update(self, logits: torch.Tensor, step: int, token_positions: Optional[torch.Tensor] = None) -> None:
        """
        新しいステップの情報を追加し、JSDを計算

        Args:
            logits: (batch_size, seq_len, vocab_size) の logits テンソル
            step: 現在のステップ番号
            token_positions: 評価対象のトークン位置マスク
        """
        # 指定された位置のトークンのみを評価
        if token_positions is not None:
            eval_logits = logits[token_positions]
        else:
            eval_logits = logits.view(-1, logits.size(-1))

        # 温度調整されたsoftmaxで確率分布を計算
        current_probs = F.softmax(eval_logits / self.temperature, dim=-1)

        if self.previous_probs is not None:
            # 各トークン位置についてJSDを計算
            jsd_values = []

            for i in range(current_probs.size(0)):
                prev_dist = self.previous_probs[i].detach().cpu().numpy()
                curr_dist = current_probs[i].detach().cpu().numpy()

                # 数値安定性のため小さな値を追加
                prev_dist = prev_dist + 1e-10
                curr_dist = curr_dist + 1e-10

                # 正規化
                prev_dist = prev_dist / prev_dist.sum()
                curr_dist = curr_dist / curr_dist.sum()

                # Jensen-Shannon Divergence を計算
                jsd = jensenshannon(prev_dist, curr_dist)
                jsd_values.append(jsd)

            # 平均JSDを記録
            avg_jsd = np.mean(jsd_values)
            self.total_jsd += avg_jsd
            self.total_comparisons += 1

        self.previous_probs = current_probs.clone()
        self.num_tokens = current_probs.size(0)
        self.num_steps = step + 1

    def value(self) -> float:
        """
        確率的GTSスコアを計算

        Returns:
            0.0-1.0 の範囲のスコア（1.0が最も安定）
        """
        if self.total_comparisons == 0:
            return 1.0  # 比較データがない場合は完全に安定と仮定

        # 平均JSDを計算（JSDは0-1の範囲）
        avg_jsd = self.total_jsd / self.total_comparisons

        # GTSスコア = 1 - JSD (JSDが小さいほど安定)
        stability = 1.0 - avg_jsd

        return max(0.0, min(1.0, stability))


def create_gts_metric(metric_type: str, **kwargs) -> BaseGTSMetric:
    """
    指定されたタイプのGTSメトリクスを作成するファクトリ関数

    Args:
        metric_type: 'basic', 'semantic', 'probabilistic' のいずれか
        **kwargs: 各メトリクス固有の引数

    Returns:
        初期化されたGTSメトリクスインスタンス

    Raises:
        ValueError: 未知のメトリクスタイプが指定された場合
    """
    if metric_type.lower() == 'basic':
        return BasicGTS(**kwargs)
    elif metric_type.lower() == 'semantic':
        return SemanticGTS(**kwargs)
    elif metric_type.lower() == 'probabilistic':
        return ProbabilisticGTS(**kwargs)
    else:
        raise ValueError(f"Unknown metric type: {metric_type}. "
                         f"Available types: 'basic', 'semantic', 'probabilistic'")


def evaluate_trajectory_stability(
    logits_sequence: List[torch.Tensor],
    token_positions: Optional[torch.Tensor] = None,
    metric_type: str = 'probabilistic',
    **metric_kwargs
) -> Dict[str, float]:
    """
    軌跡データからGTSスコアを一括計算するユーティリティ関数

    Args:
        logits_sequence: ステップごとのlogitsテンソルのリスト
        token_positions: 評価対象のトークン位置マスク
        metric_type: 使用するメトリクスタイプ
        **metric_kwargs: メトリクス固有の引数

    Returns:
        計算されたGTSスコアと関連統計を含む辞書
    """
    metric = create_gts_metric(metric_type, **metric_kwargs)

    for step, logits in enumerate(logits_sequence):
        metric.update(logits, step, token_positions)

    gts_score = metric.value()

    return {
        'gts_score': gts_score,
        'metric_type': metric_type,
        'num_steps': len(logits_sequence),
        'num_tokens': metric.num_tokens
    }
