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
from collections import deque
import math


class AdaptiveInferenceScheduler:
    """
    Self-Correcting Adaptive Inference Scheduler for Diffusion LLMs.

    This class implements dynamic block sizing and adaptive confidence thresholding
    based on real-time model confidence and prediction entropy.
    """

    def __init__(
        self,
        min_block_size: int = 8,
        max_block_size: int = 64,
        base_confidence_threshold: float = 0.8,
        min_threshold: float = 0.7,
        max_threshold: float = 0.95,
        confidence_window: int = 5,
        adaptation_sensitivity: float = 0.1,   # より積極的な適応（0.02 -> 0.1）
        entropy_threshold_high: float = 0.8,   # さらに低い閾値（1.0 -> 0.8）
        entropy_threshold_low: float = 0.3,    # さらに低い閾値（0.5 -> 0.3）
        scale_up_factor: float = 1.6,         # さらに積極的（1.5 -> 1.6）
        scale_down_factor: float = 0.5,       # さらに積極的（0.6 -> 0.5）
        safety_factor: float = 1.15,          # より大きな調整（1.1 -> 1.15）
        efficiency_factor: float = 0.85       # より大きな調整（0.9 -> 0.85）
    ):
        """
        アダプティブスケジューラーの初期化

        Args:
            min_block_size: 最小ブロックサイズ
            max_block_size: 最大ブロックサイズ
            base_confidence_threshold: 基本信頼度閾値
            min_threshold: 最小信頼度閾値
            max_threshold: 最大信頼度閾値
            confidence_window: 信頼度計算ウィンドウサイズ
            adaptation_sensitivity: 適応感度
            entropy_threshold_high: 高エントロピー閾値
            entropy_threshold_low: 低エントロピー閾値
            scale_up_factor: ブロックサイズ拡大係数
            scale_down_factor: ブロックサイズ縮小係数
            safety_factor: 安全性係数（閾値上昇）
            efficiency_factor: 効率性係数（閾値下降）
        """
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.base_confidence_threshold = base_confidence_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.confidence_window = confidence_window
        self.adaptation_sensitivity = adaptation_sensitivity
        self.entropy_threshold_high = entropy_threshold_high
        self.entropy_threshold_low = entropy_threshold_low
        self.scale_up_factor = scale_up_factor
        self.scale_down_factor = scale_down_factor
        self.safety_factor = safety_factor
        self.efficiency_factor = efficiency_factor

        # 適応状態の初期化
        self.reset_state()

    def reset_state(self):
        """適応状態をリセット"""
        # 初期ブロックサイズは min_block_size の1.5倍程度に設定（より保守的に開始）
        self.current_block_size = max(self.min_block_size,
                                      min(self.min_block_size * 2, self.max_block_size))
        self.current_threshold = self.base_confidence_threshold
        self.confidence_history = deque(maxlen=self.confidence_window)
        self.entropy_history = deque(maxlen=self.confidence_window)
        self.adaptation_count = 0
        self.total_blocks = 0

    def calculate_entropy(self, logits: torch.Tensor, mask_index: torch.Tensor) -> float:
        """
        予測エントロピーを計算

        Args:
            logits: モデルの出力ロジット
            mask_index: マスクされた位置のインデックス

        Returns:
            平均エントロピー
        """
        # None チェック
        if logits is None or mask_index is None:
            return 0.0

        # 形状の整合性を確認
        if logits.shape[:2] != mask_index.shape:
            # 最小の次元に合わせる
            min_seq_len = min(logits.shape[1], mask_index.shape[1])
            logits = logits[:, :min_seq_len]
            mask_index = mask_index[:, :min_seq_len]

        # マスクされた位置のみのロジットを取得
        masked_logits = logits[mask_index]
        if masked_logits.numel() == 0:
            return 0.0

        # ソフトマックス確率を計算
        probs = F.softmax(masked_logits.to(torch.float64), dim=-1)

        # エントロピーを計算（対数の底は2）
        log_probs = torch.log2(probs + 1e-10)  # 数値安定性のため小さな値を追加
        entropy = -(probs * log_probs).sum(dim=-1)

        return entropy.mean().item()

    def calculate_confidence_metrics(self, logits: torch.Tensor, tokens: torch.Tensor,
                                     mask_index: torch.Tensor) -> Tuple[float, float]:
        """
        信頼度メトリクスを計算

        Args:
            logits: モデルの出力ロジット（ブロック分のみ）
            tokens: 生成されたトークン（ブロック分のみ）
            mask_index: マスクされた位置のインデックス（ブロック分のみ）

        Returns:
            (平均信頼度, エントロピー)
        """
        # None チェック
        if logits is None or tokens is None or mask_index is None:
            return 0.0, 0.0

        # エントロピーを計算
        entropy = self.calculate_entropy(logits, mask_index)

        # 信頼度を計算
        if not mask_index.any():
            return 0.0, entropy

        # 形状の整合性を確保
        min_seq_len = min(
            logits.shape[1], tokens.shape[1], mask_index.shape[1])
        logits_trimmed = logits[:, :min_seq_len]
        tokens_trimmed = tokens[:, :min_seq_len]
        mask_trimmed = mask_index[:, :min_seq_len]

        # ソフトマックス確率を計算
        probs = F.softmax(logits_trimmed.to(torch.float64), dim=-1)

        # トークンの確率を取得
        token_probs = torch.gather(
            probs, dim=-1, index=tokens_trimmed.unsqueeze(-1)).squeeze(-1)

        # マスクされた位置の信頼度のみを考慮
        masked_confidences = token_probs[mask_trimmed]
        avg_confidence = masked_confidences.mean().item(
        ) if masked_confidences.numel() > 0 else 0.0

        return avg_confidence, entropy

    def adapt_block_size(self, avg_confidence: float) -> int:
        """
        信頼度に基づいてブロックサイズを動的調整

        Args:
            avg_confidence: 現在のブロックの平均信頼度

        Returns:
            次のブロックサイズ
        """
        # 信頼度履歴を更新
        self.confidence_history.append(avg_confidence)

        # ウィンドウ内の信頼度傾向を分析（最低1回の観測で開始）
        if len(self.confidence_history) < 1:
            return self.current_block_size

        # 最近の信頼度を使用（単一の値でも適応可能）
        recent_avg = avg_confidence  # 即座に反応

        # 高信頼度: ブロックサイズを拡大
        high_threshold = self.current_threshold + self.adaptation_sensitivity
        low_threshold = self.current_threshold - self.adaptation_sensitivity

        old_size = self.current_block_size

        if recent_avg > high_threshold:
            new_size = min(self.max_block_size,
                           int(self.current_block_size * self.scale_up_factor))
        elif recent_avg < low_threshold:
            new_size = max(self.min_block_size,
                           int(self.current_block_size * self.scale_down_factor))
        else:
            new_size = self.current_block_size

        self.current_block_size = new_size
        return new_size

    def adapt_threshold(self, entropy: float) -> float:
        """
        エントロピーに基づいて信頼度閾値を動的調整

        Args:
            entropy: 現在の予測エントロピー

        Returns:
            調整された信頼度閾値
        """
        # エントロピー履歴を更新
        self.entropy_history.append(entropy)

        # 高エントロピー（不確実）: より慎重に（閾値上昇）
        if entropy > self.entropy_threshold_high:
            new_threshold = min(self.max_threshold,
                                self.current_threshold * self.safety_factor)
        # 低エントロピー（確実）: より積極的に（閾値下降）
        elif entropy < self.entropy_threshold_low:
            new_threshold = max(self.min_threshold,
                                self.current_threshold * self.efficiency_factor)
        else:
            # 中程度のエントロピー: 閾値を基本値に向けて緩やかに調整
            target = self.base_confidence_threshold
            adjustment = (target - self.current_threshold) * 0.1
            new_threshold = self.current_threshold + adjustment

        self.current_threshold = new_threshold
        return new_threshold

    def should_adapt(self, step: int, total_steps: int) -> bool:
        """
        適応を実行すべきかどうかを判定

        Args:
            step: 現在のステップ
            total_steps: 総ステップ数

        Returns:
            適応実行フラグ
        """
        # 毎ステップ適応を許可（最初のステップからも適応）
        return True

    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """
        適応メトリクスを取得

        Returns:
            適応メトリクスの辞書
        """
        return {
            'current_block_size': self.current_block_size,
            'current_threshold': self.current_threshold,
            'adaptation_count': self.adaptation_count,
            'total_blocks': self.total_blocks,
            'avg_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0.0,
            'avg_entropy': np.mean(list(self.entropy_history)) if self.entropy_history else 0.0,
            'confidence_std': np.std(list(self.confidence_history)) if len(self.confidence_history) > 1 else 0.0,
            'entropy_std': np.std(list(self.entropy_history)) if len(self.entropy_history) > 1 else 0.0
        }

    def step(self, logits: torch.Tensor, tokens: torch.Tensor, mask_index: torch.Tensor,
             step_num: int, total_steps: int) -> Tuple[int, float, Dict[str, float]]:
        """
        適応ステップを実行

        Args:
            logits: モデルの出力ロジット
            tokens: 生成されたトークン
            mask_index: マスクされた位置のインデックス
            step_num: 現在のステップ番号
            total_steps: 総ステップ数

        Returns:
            (次のブロックサイズ, 調整された閾値, メトリクス)
        """
        # 信頼度とエントロピーを計算
        avg_confidence, entropy = self.calculate_confidence_metrics(
            logits, tokens, mask_index)

        # 現在の状態を保存（適応検出用）
        old_block_size = self.current_block_size
        old_threshold = self.current_threshold

        # 適応を実行すべきかチェック
        if self.should_adapt(step_num, total_steps):
            # ブロックサイズを適応
            next_block_size = self.adapt_block_size(avg_confidence)

            # 閾値を適応
            adapted_threshold = self.adapt_threshold(entropy)

            # 適応が実際に発生したかチェック（より緩い条件）
            if (next_block_size != old_block_size or
                    abs(adapted_threshold - old_threshold) > 0.001):
                self.adaptation_count += 1
        else:
            next_block_size = self.current_block_size
            adapted_threshold = self.current_threshold

        self.total_blocks += 1

        # ステップメトリクス
        step_metrics = {
            'confidence': avg_confidence,
            'entropy': entropy,
            'block_size': next_block_size,
            'threshold': adapted_threshold,
            'adapted': (next_block_size != old_block_size or
                        abs(adapted_threshold - old_threshold) > 0.001)
        }

        return next_block_size, adapted_threshold, step_metrics


class BlockSizeController:
    """
    Dynamic Block Size Controller
    ブロックサイズの動的制御を担当するコンポーネント
    """

    def __init__(self, min_size: int = 8, max_size: int = 64,
                 scale_factor: float = 1.2, adaptation_rate: float = 0.2):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_factor = scale_factor
        self.adaptation_rate = adaptation_rate
        self.size_history = deque(maxlen=10)

    def adjust_size(self, current_size: int, confidence: float,
                    threshold: float) -> int:
        """信頼度に基づいてブロックサイズを調整"""
        if confidence > threshold + 0.1:
            # 高信頼度: サイズ拡大
            new_size = min(self.max_size, int(
                current_size * self.scale_factor))
        elif confidence < threshold - 0.1:
            # 低信頼度: サイズ縮小
            new_size = max(self.min_size, int(
                current_size / self.scale_factor))
        else:
            new_size = current_size

        self.size_history.append(new_size)
        return new_size


class ThresholdController:
    """
    Adaptive Threshold Controller
    信頼度閾値の動的制御を担当するコンポーネント
    """

    def __init__(self, base_threshold: float = 0.8, min_threshold: float = 0.7,
                 max_threshold: float = 0.95, adaptation_rate: float = 0.1):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_rate = adaptation_rate
        self.threshold_history = deque(maxlen=10)

    def adjust_threshold(self, current_threshold: float, entropy: float,
                         high_entropy: float = 2.0, low_entropy: float = 0.5) -> float:
        """エントロピーに基づいて閾値を調整"""
        if entropy > high_entropy:
            # 高エントロピー: 閾値上昇（より慎重に）
            adjustment = self.adaptation_rate * \
                (entropy - high_entropy) / high_entropy
            new_threshold = min(self.max_threshold,
                                current_threshold + adjustment)
        elif entropy < low_entropy:
            # 低エントロピー: 閾値下降（より積極的に）
            adjustment = self.adaptation_rate * \
                (low_entropy - entropy) / low_entropy
            new_threshold = max(self.min_threshold,
                                current_threshold - adjustment)
        else:
            # 基本閾値に向けて緩やかに調整
            new_threshold = current_threshold + \
                (self.base_threshold - current_threshold) * 0.05

        self.threshold_history.append(new_threshold)
        return new_threshold
