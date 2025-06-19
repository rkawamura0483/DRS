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
from typing import Tuple, Dict, Any
from collections import deque
from enum import Enum


class InferenceMode(Enum):
    """
    生成モードを定義するEnum
    """
    HIGH_EFFICIENCY = "High Efficiency"
    HIGH_QUALITY = "High Quality"


class AdaptiveInferenceScheduler:
    """
    モード切り替えに基づく適応的スケジューラー。
    高効率モードと高品質モードを動的に切り替える。
    """

    def __init__(
        self,
        # モード遷移に関するパラメータ
        to_quality_threshold: float = 0.80,
        to_efficiency_threshold: float = 0.95,
        confidence_window_size: int = 2,
        # 各モードのパラメータ
        high_efficiency_params: Dict[str, Any] = None,
        high_quality_params: Dict[str, Any] = None,
    ):
        """
        モードベースのスケジューラーを初期化

        Args:
            to_quality_threshold: 高品質モードに遷移するための信頼度の閾値
            to_efficiency_threshold: 高効率モードに遷移するための信頼度の閾値
            confidence_window_size: 信頼度を平均化するウィンドウサイズ
            high_efficiency_params: 高効率モードのパラメータ辞書
            high_quality_params: 高品質モードのパラメータ辞書
        """
        self.to_quality_threshold = to_quality_threshold
        self.to_efficiency_threshold = to_efficiency_threshold

        # モードごとのパラメータ設定
        self.mode_params = {
            InferenceMode.HIGH_EFFICIENCY: high_efficiency_params or
            {'block_size': 32, 'threshold': 0.75},
            InferenceMode.HIGH_QUALITY: high_quality_params or
            {'block_size': 8, 'threshold': 0.95}
        }

        # 適応状態の初期化
        self.confidence_history = deque(maxlen=confidence_window_size)
        self.current_mode = InferenceMode.HIGH_EFFICIENCY
        self.adaptation_count = 0
        self.total_blocks = 0

    @property
    def current_block_size(self) -> int:
        """現在のモードに基づいたブロックサイズを返す"""
        return self.mode_params[self.current_mode]['block_size']

    @property
    def current_threshold(self) -> float:
        """現在のモードに基づいた信頼度閾値を返す"""
        return self.mode_params[self.current_mode]['threshold']

    def adapt_mode(self, block_confidence: float) -> bool:
        """
        信頼度に基づいてモードを適応させる。

        Args:
            block_confidence: 直近のブロックの平均信頼度

        Returns:
            モードが変更されたかどうか
        """
        self.total_blocks += 1
        self.confidence_history.append(block_confidence)

        if len(self.confidence_history) < self.confidence_history.maxlen:
            return False  # 履歴が溜まるまで適応しない

        avg_confidence = np.mean(list(self.confidence_history))
        original_mode = self.current_mode

        # 高品質モードへの遷移ロジック
        if self.current_mode == InferenceMode.HIGH_EFFICIENCY and avg_confidence < self.to_quality_threshold:
            self.current_mode = InferenceMode.HIGH_QUALITY

        # 高効率モードへの遷移ロジック
        elif self.current_mode == InferenceMode.HIGH_QUALITY and avg_confidence > self.to_efficiency_threshold:
            self.current_mode = InferenceMode.HIGH_EFFICIENCY

        mode_changed = (original_mode != self.current_mode)
        if mode_changed:
            self.adaptation_count += 1

        return mode_changed

    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """
        適応に関するメトリクスを返す
        """
        return {
            'adaptation_count': self.adaptation_count,
            'total_blocks': self.total_blocks,
            'current_mode': self.current_mode.value,
            'current_block_size': self.current_block_size,
            'current_threshold': self.current_threshold
        }

    # calculate_entropyは他の場所で使われる可能性があるため残す
    def calculate_entropy(self, logits: torch.Tensor, mask_index: torch.Tensor) -> float:
        """
        予測エントロピーを計算
        """
        if logits is None or mask_index is None:
            return 0.0

        if logits.shape[:2] != mask_index.shape:
            min_seq_len = min(logits.shape[1], mask_index.shape[1])
            logits = logits[:, :min_seq_len]
            mask_index = mask_index[:, :min_seq_len]

        masked_logits = logits[mask_index]
        if masked_logits.numel() == 0:
            return 0.0

        probs = F.softmax(masked_logits.to(torch.float64), dim=-1)
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)

        return entropy.mean().item()
