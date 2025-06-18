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
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
from enum import Enum


class CacheTier(Enum):
    """キャッシュ階層の定義"""
    FROZEN = "frozen"    # Tier 1: プロンプトトークン（一度設定したら更新なし）
    STABLE = "stable"    # Tier 2: 高信頼度生成ブロック（低頻度更新）
    ACTIVE = "active"    # Tier 3: 最近/不確実ブロック（毎ステップ更新）


class TieredCacheManager:
    """
    Tiered Cache Management System

    三階層キャッシュシステムを実装:
    - Tier 1 (Frozen): プロンプトトークン用、一度だけ計算
    - Tier 2 (Stable): 高信頼度ブロック用、稀に更新
    - Tier 3 (Active): 不確実ブロック用、毎ステップ更新
    """

    def __init__(
        self,
        tier2_stability_threshold: float = 0.85,
        tier2_update_interval: int = 3,
        memory_efficiency_mode: bool = True,
        max_stable_blocks: int = 32,
        confidence_memory_window: int = 5
    ):
        """
        階層キャッシュマネージャーの初期化

        Args:
            tier2_stability_threshold: Tier2への昇格に必要な信頼度閾値
            tier2_update_interval: Tier2の更新間隔
            memory_efficiency_mode: メモリ効率モード
            max_stable_blocks: 最大安定ブロック数
            confidence_memory_window: 信頼度メモリウィンドウ
        """
        self.tier2_stability_threshold = tier2_stability_threshold
        self.tier2_update_interval = tier2_update_interval
        self.memory_efficiency_mode = memory_efficiency_mode
        self.max_stable_blocks = max_stable_blocks
        self.confidence_memory_window = confidence_memory_window

        # キャッシュ状態の初期化
        self.reset_cache()

    def reset_cache(self):
        """キャッシュ状態をリセット"""
        # 各階層のキャッシュデータ
        self.tier1_cache = None  # Frozen: past_key_values for prompt
        self.tier2_cache = {}    # Stable: {block_id: past_key_values}
        self.tier3_cache = None  # Active: current past_key_values

        # ブロック管理
        self.block_tiers = {}     # block_id -> CacheTier
        self.block_confidences = defaultdict(
            list)  # block_id -> [confidence_scores]
        self.block_positions = {}  # block_id -> (start_pos, end_pos)
        self.stable_blocks = set()  # 安定ブロックのIDセット

        # 更新カウンタ
        self.update_counters = defaultdict(int)  # block_id -> update_count
        self.total_updates = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # プロンプト情報
        self.prompt_length = 0
        self.is_prompt_cached = False

    def set_prompt_cache(self, prompt_length: int, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        プロンプト用のTier1キャッシュを設定

        Args:
            prompt_length: プロンプトの長さ
            past_key_values: プロンプトに対するkey-valueキャッシュ
        """
        self.prompt_length = prompt_length
        self.tier1_cache = self._deep_copy_cache(past_key_values)
        self.is_prompt_cached = True

        # プロンプトキャッシュ設定時にヒットとしてカウント
        self.cache_hits += 1

        print(f"Tier1 (Frozen): プロンプトキャッシュ設定完了 (長さ: {prompt_length})")

    def classify_block(self, block_id: int, confidence_scores: torch.Tensor) -> CacheTier:
        """
        ブロックの信頼度に基づいて階層を分類

        Args:
            block_id: ブロックID
            confidence_scores: ブロック内トークンの信頼度スコア

        Returns:
            分類された階層
        """
        # 信頼度履歴を更新
        avg_confidence = confidence_scores.mean().item()
        self.block_confidences[block_id].append(avg_confidence)

        # ウィンドウサイズに合わせて履歴を管理
        if len(self.block_confidences[block_id]) > self.confidence_memory_window:
            self.block_confidences[block_id] = self.block_confidences[block_id][-self.confidence_memory_window:]

        # 安定性の判定
        recent_confidences = self.block_confidences[block_id]
        if len(recent_confidences) >= 3:  # 最低3回の観測が必要
            stable_confidence = np.mean(
                recent_confidences) > self.tier2_stability_threshold
            confidence_stable = np.std(recent_confidences) < 0.05  # 信頼度の変動が小さい

            if stable_confidence and confidence_stable and len(self.stable_blocks) < self.max_stable_blocks:
                return CacheTier.STABLE

        return CacheTier.ACTIVE

    def should_update_tier2(self, block_id: int) -> bool:
        """
        Tier2ブロックを更新すべきかどうかを判定

        Args:
            block_id: ブロックID

        Returns:
            更新フラグ
        """
        if block_id not in self.update_counters:
            return True

        return self.update_counters[block_id] % self.tier2_update_interval == 0

    def update_cache(self, block_id: int, start_pos: int, end_pos: int,
                     past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
                     confidence_scores: torch.Tensor) -> CacheTier:
        """
        ブロックのキャッシュを更新

        Args:
            block_id: ブロックID
            start_pos: ブロック開始位置
            end_pos: ブロック終了位置
            past_key_values: key-valueキャッシュ
            confidence_scores: 信頼度スコア

        Returns:
            使用された階層
        """
        # ブロック位置を記録
        self.block_positions[block_id] = (start_pos, end_pos)

        # 階層を分類
        tier = self.classify_block(block_id, confidence_scores)
        self.block_tiers[block_id] = tier

        if tier == CacheTier.STABLE:
            # Tier2: 安定ブロック
            if self.should_update_tier2(block_id):
                # プロンプト部分を除いてキャッシュを保存
                cache_to_save = self._trim_cache_to_position(
                    past_key_values, start_pos)
                self.tier2_cache[block_id] = self._deep_copy_cache(
                    cache_to_save)
                self.stable_blocks.add(block_id)
                print(f"Tier2 (Stable): ブロック {block_id} キャッシュ更新")
            else:
                # 再利用時はヒットとしてカウント
                self.cache_hits += 1
                print(f"Tier2 (Stable): ブロック {block_id} キャッシュ再利用")

        elif tier == CacheTier.ACTIVE:
            # Tier3: アクティブブロック
            self.tier3_cache = self._deep_copy_cache(past_key_values)
            self.cache_misses += 1
            print(f"Tier3 (Active): ブロック {block_id} キャッシュ更新")

        self.update_counters[block_id] += 1
        self.total_updates += 1

        return tier

    def get_cache_for_block(self, block_id: int) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        ブロック用のキャッシュを取得

        Args:
            block_id: ブロックID

        Returns:
            key-valueキャッシュ、またはNone
        """
        if block_id not in self.block_tiers:
            return None

        tier = self.block_tiers[block_id]

        if tier == CacheTier.STABLE and block_id in self.tier2_cache:
            # Tier2から取得
            base_cache = self.tier2_cache[block_id]
            # プロンプトキャッシュと結合
            if self.tier1_cache is not None:
                return self._combine_caches(self.tier1_cache, base_cache)
            return base_cache

        elif tier == CacheTier.ACTIVE and self.tier3_cache is not None:
            # Tier3から取得
            return self.tier3_cache

        return None

    def get_base_cache(self) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        基本キャッシュ（プロンプト）を取得

        Returns:
            プロンプト用key-valueキャッシュ
        """
        return self.tier1_cache

    def _deep_copy_cache(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """キャッシュのディープコピーを作成"""
        if past_key_values is None:
            return None

        copied_cache = []
        for layer_cache in past_key_values:
            if layer_cache is None:
                copied_cache.append(None)
            else:
                copied_layer = tuple(tensor.clone() for tensor in layer_cache)
                copied_cache.append(copied_layer)
        return copied_cache

    def _trim_cache_to_position(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
                                position: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        指定された位置までキャッシュをトリム

        Args:
            past_key_values: 元のキャッシュ
            position: トリム位置

        Returns:
            トリムされたキャッシュ
        """
        if past_key_values is None:
            return None

        trimmed_cache = []
        for layer_cache in past_key_values:
            if layer_cache is None:
                trimmed_cache.append(None)
            else:
                # key, valueの次元: (batch_size, num_heads, seq_len, head_dim)
                key, value = layer_cache
                trimmed_key = key[:, :, :position]
                trimmed_value = value[:, :, :position]
                trimmed_cache.append((trimmed_key, trimmed_value))
        return trimmed_cache

    def _combine_caches(self, cache1: List[Tuple[torch.Tensor, torch.Tensor]],
                        cache2: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        二つのキャッシュを結合

        Args:
            cache1: 最初のキャッシュ
            cache2: 二番目のキャッシュ

        Returns:
            結合されたキャッシュ
        """
        if cache1 is None:
            return cache2
        if cache2 is None:
            return cache1

        combined_cache = []
        for layer1, layer2 in zip(cache1, cache2):
            if layer1 is None or layer2 is None:
                combined_cache.append(layer1 or layer2)
            else:
                key1, value1 = layer1
                key2, value2 = layer2
                # 次元2（seq_len）で結合
                combined_key = torch.cat([key1, key2], dim=2)
                combined_value = torch.cat([value1, value2], dim=2)
                combined_cache.append((combined_key, combined_value))
        return combined_cache

    def promote_to_stable(self, block_id: int):
        """
        ブロックをTier2（安定）に昇格

        Args:
            block_id: ブロックID
        """
        if block_id in self.block_tiers:
            self.block_tiers[block_id] = CacheTier.STABLE
            self.stable_blocks.add(block_id)
            print(f"ブロック {block_id} をTier2に昇格")

    def demote_to_active(self, block_id: int):
        """
        ブロックをTier3（アクティブ）に降格

        Args:
            block_id: ブロックID
        """
        if block_id in self.block_tiers:
            self.block_tiers[block_id] = CacheTier.ACTIVE
            if block_id in self.stable_blocks:
                self.stable_blocks.remove(block_id)
            if block_id in self.tier2_cache:
                del self.tier2_cache[block_id]
            print(f"ブロック {block_id} をTier3に降格")

    def cleanup_old_caches(self, current_blocks: List[int]):
        """
        使用されなくなったキャッシュをクリーンアップ

        Args:
            current_blocks: 現在アクティブなブロックIDのリスト
        """
        if not self.memory_efficiency_mode:
            return

        # 現在使用されていないTier2キャッシュを削除
        unused_blocks = set(self.tier2_cache.keys()) - set(current_blocks)
        for block_id in unused_blocks:
            if block_id in self.tier2_cache:
                del self.tier2_cache[block_id]
                self.stable_blocks.discard(block_id)
                print(f"未使用キャッシュを削除: ブロック {block_id}")

    def get_cache_efficiency_metrics(self) -> Dict[str, Any]:
        """
        キャッシュ効率メトリクスを取得

        Returns:
            キャッシュ効率メトリクス
        """
        total_cache_ops = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0

        # メモリ使用量の推定
        tier1_memory = self._estimate_cache_memory(
            self.tier1_cache) if self.tier1_cache else 0
        tier2_memory = sum(self._estimate_cache_memory(cache)
                           for cache in self.tier2_cache.values())
        tier3_memory = self._estimate_cache_memory(
            self.tier3_cache) if self.tier3_cache else 0

        return {
            'cache_hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_updates': self.total_updates,
            'tier1_blocks': 1 if self.is_prompt_cached else 0,
            'tier2_blocks': len(self.stable_blocks),
            'tier3_active': 1 if self.tier3_cache is not None else 0,
            'memory_usage': {
                'tier1_mb': tier1_memory / (1024 * 1024),
                'tier2_mb': tier2_memory / (1024 * 1024),
                'tier3_mb': tier3_memory / (1024 * 1024),
                'total_mb': (tier1_memory + tier2_memory + tier3_memory) / (1024 * 1024)
            }
        }

    def _estimate_cache_memory(self, cache: List[Tuple[torch.Tensor, torch.Tensor]]) -> int:
        """
        キャッシュのメモリ使用量を推定（バイト）

        Args:
            cache: key-valueキャッシュ

        Returns:
            推定メモリ使用量（バイト）
        """
        if cache is None:
            return 0

        total_bytes = 0
        for layer_cache in cache:
            if layer_cache is not None:
                for tensor in layer_cache:
                    total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes

    def print_cache_status(self):
        """キャッシュ状態を出力"""
        metrics = self.get_cache_efficiency_metrics()
        print(f"\n=== キャッシュ状態 ===")
        print(f"ヒット率: {metrics['cache_hit_rate']:.2%}")
        print(f"Tier1 (Frozen): {metrics['tier1_blocks']} ブロック")
        print(f"Tier2 (Stable): {metrics['tier2_blocks']} ブロック")
        print(f"Tier3 (Active): {metrics['tier3_active']} ブロック")
        print(f"総メモリ使用量: {metrics['memory_usage']['total_mb']:.1f} MB")
        print(f"総更新回数: {metrics['total_updates']}")
