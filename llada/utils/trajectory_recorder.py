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
軌跡記録ユーティリティ

diffusion LLMの生成過程における中間状態を記録し、
GTSメトリクス計算用のデータを提供します。
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class TrajectoryRecorder:
    """
    生成軌跡を記録するクラス

    各ステップでのlogitsと確率分布を保存し、
    GTSメトリクス計算に必要なデータを提供します。
    """

    def __init__(self,
                 max_steps: int = 100,
                 record_mode: str = 'logits',
                 device: Optional[torch.device] = None):
        """
        軌跡記録器の初期化

        Args:
            max_steps: 記録する最大ステップ数
            record_mode: 記録モード ('logits', 'probs', 'both')
            device: 使用するデバイス
        """
        self.max_steps = max_steps
        self.record_mode = record_mode
        self.device = device if device is not None else torch.device('cpu')

        # 記録データの初期化
        self.reset()

    def reset(self):
        """記録データをリセット"""
        self.step_count = 0
        self.logits_history = []    # List[torch.Tensor]
        self.probs_history = []     # List[torch.Tensor]
        self.argmax_history = []    # List[torch.Tensor]
        self.metadata_history = []  # List[Dict]

        # トークン別の履歴（辞書形式で効率的にアクセス）
        self.token_logits = defaultdict(list)
        self.token_probs = defaultdict(list)
        self.token_argmax = defaultdict(list)

    def record_step(self,
                    logits: torch.Tensor,
                    step: int,
                    token_positions: Optional[torch.Tensor] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        1ステップの情報を記録

        Args:
            logits: (batch_size, seq_len, vocab_size) のlogitsテンソル
            step: ステップ番号
            token_positions: 記録対象のトークン位置マスク
            metadata: 追加のメタデータ
        """
        if self.step_count >= self.max_steps:
            # 古いデータを削除して新しいデータを追加（リングバッファ的動作）
            self._remove_oldest_step()

        # 記録対象の選択
        if token_positions is not None:
            # マスクが指定された場合は該当位置のみ
            record_logits = logits[token_positions]
        else:
            # 全位置を記録
            record_logits = logits.view(-1, logits.size(-1))

        # logitsの記録
        if self.record_mode in ['logits', 'both']:
            self.logits_history.append(
                record_logits.clone().detach().to(self.device))

        # 確率分布の計算と記録
        if self.record_mode in ['probs', 'both']:
            probs = F.softmax(record_logits, dim=-1)
            self.probs_history.append(probs.clone().detach().to(self.device))

        # argmaxの記録（常に記録）
        argmax = torch.argmax(record_logits, dim=-1)
        self.argmax_history.append(argmax.clone().detach().to(self.device))

        # メタデータの記録
        metadata = metadata or {}
        metadata.update({
            'step': step,
            'num_tokens': record_logits.size(0),
            'vocab_size': record_logits.size(1)
        })
        self.metadata_history.append(metadata)

        # トークン別履歴の更新
        self._update_token_histories(record_logits, step, token_positions)

        self.step_count += 1

    def _update_token_histories(self,
                                logits: torch.Tensor,
                                step: int,
                                token_positions: Optional[torch.Tensor] = None):
        """トークン別の履歴を更新"""
        for token_idx in range(logits.size(0)):
            # logits履歴
            if self.record_mode in ['logits', 'both']:
                self.token_logits[token_idx].append(
                    logits[token_idx].clone().detach())

            # 確率履歴
            if self.record_mode in ['probs', 'both']:
                probs = F.softmax(logits[token_idx], dim=-1)
                self.token_probs[token_idx].append(probs.clone().detach())

            # argmax履歴
            argmax = torch.argmax(logits[token_idx], dim=-1)
            self.token_argmax[token_idx].append(argmax.clone().detach())

    def _remove_oldest_step(self):
        """最古のステップデータを削除"""
        if len(self.logits_history) > 0:
            self.logits_history.pop(0)
        if len(self.probs_history) > 0:
            self.probs_history.pop(0)
        if len(self.argmax_history) > 0:
            self.argmax_history.pop(0)
        if len(self.metadata_history) > 0:
            self.metadata_history.pop(0)

        # トークン別履歴も削除
        for token_idx in self.token_logits:
            if len(self.token_logits[token_idx]) > 0:
                self.token_logits[token_idx].pop(0)
        for token_idx in self.token_probs:
            if len(self.token_probs[token_idx]) > 0:
                self.token_probs[token_idx].pop(0)
        for token_idx in self.token_argmax:
            if len(self.token_argmax[token_idx]) > 0:
                self.token_argmax[token_idx].pop(0)

        self.step_count -= 1

    def get_token_history(self, token_idx: int, data_type: str = 'argmax') -> List[torch.Tensor]:
        """
        指定されたトークンの履歴を取得

        Args:
            token_idx: トークンのインデックス
            data_type: 取得するデータタイプ ('logits', 'probs', 'argmax')

        Returns:
            指定されたトークンの履歴リスト
        """
        if data_type == 'logits':
            return self.token_logits.get(token_idx, [])
        elif data_type == 'probs':
            return self.token_probs.get(token_idx, [])
        elif data_type == 'argmax':
            return self.token_argmax.get(token_idx, [])
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def get_full_history(self, data_type: str = 'logits') -> List[torch.Tensor]:
        """
        全ステップの履歴を取得

        Args:
            data_type: 取得するデータタイプ ('logits', 'probs', 'argmax')

        Returns:
            全ステップの履歴リスト
        """
        if data_type == 'logits':
            return self.logits_history
        elif data_type == 'probs':
            return self.probs_history
        elif data_type == 'argmax':
            return self.argmax_history
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

    def get_stability_data(self) -> Dict[str, Any]:
        """
        GTSメトリクス計算用の安定性データを取得

        Returns:
            安定性分析用のデータ辞書
        """
        if len(self.argmax_history) < 2:
            return {
                'has_sufficient_data': False,
                'num_steps': len(self.argmax_history),
                'message': '安定性計算には最低2ステップが必要です'
            }

        # 基本統計
        num_steps = len(self.argmax_history)
        num_tokens = self.argmax_history[0].size(0) if num_steps > 0 else 0

        # フリップ統計の計算
        total_flips = 0
        step_flips = []

        for i in range(1, num_steps):
            flips = (self.argmax_history[i] !=
                     self.argmax_history[i-1]).sum().item()
            total_flips += flips
            step_flips.append(flips)

        # トークン別の不安定性
        token_instability = {}
        for token_idx in range(num_tokens):
            token_flips = 0
            for i in range(1, num_steps):
                if self.argmax_history[i][token_idx] != self.argmax_history[i-1][token_idx]:
                    token_flips += 1
            token_instability[token_idx] = token_flips

        return {
            'has_sufficient_data': True,
            'num_steps': num_steps,
            'num_tokens': num_tokens,
            'total_flips': total_flips,
            'step_flips': step_flips,
            'avg_flips_per_step': total_flips / max(1, num_steps - 1),
            'token_instability': token_instability,
            'most_unstable_tokens': sorted(token_instability.items(),
                                           key=lambda x: x[1], reverse=True)[:5],
            'basic_gts_score': 1.0 - (total_flips / max(1, num_tokens * (num_steps - 1)))
        }

    def get_logits_sequence_for_gts(self,
                                    token_positions: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        GTSメトリクス計算用のlogits系列を取得

        Args:
            token_positions: 対象トークンのマスク

        Returns:
            GTSメトリクスに直接渡せるlogits系列
        """
        if not self.logits_history:
            return []

        # 元のバッチ形状に復元
        formatted_logits = []
        for step_logits in self.logits_history:
            # (num_selected_tokens, vocab_size) を (1, num_tokens, vocab_size) に変換
            if step_logits.dim() == 2:
                formatted = step_logits.unsqueeze(0)  # バッチ次元を追加
            else:
                formatted = step_logits
            formatted_logits.append(formatted)

        return formatted_logits

    def analyze_convergence(self, window_size: int = 3) -> Dict[str, Any]:
        """
        軌跡の収束性を分析

        Args:
            window_size: 分析ウィンドウのサイズ

        Returns:
            収束性分析の結果
        """
        if len(self.argmax_history) < window_size:
            return {'converged': False, 'reason': 'insufficient_data'}

        # 最近のwindow_size個のステップでフリップがあったかチェック
        recent_steps = self.argmax_history[-window_size:]
        has_flips = False

        for i in range(1, len(recent_steps)):
            if (recent_steps[i] != recent_steps[i-1]).any():
                has_flips = True
                break

        return {
            'converged': not has_flips,
            'window_size': window_size,
            'steps_analyzed': len(recent_steps),
            'total_steps': len(self.argmax_history)
        }

    def get_summary(self) -> Dict[str, Any]:
        """記録された軌跡の要約を取得"""
        stability_data = self.get_stability_data()
        convergence_data = self.analyze_convergence()

        return {
            'total_steps': self.step_count,
            'record_mode': self.record_mode,
            'stability': stability_data,
            'convergence': convergence_data,
            'memory_usage': {
                'logits_tensors': len(self.logits_history),
                'probs_tensors': len(self.probs_history),
                'argmax_tensors': len(self.argmax_history)
            }
        }
