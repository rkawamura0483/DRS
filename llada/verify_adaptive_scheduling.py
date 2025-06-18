#!/usr/bin/env python3
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
Self-Correcting Adaptive Inference Scheduling 実装検証スクリプト

このスクリプトは、アダプティブスケジューリングシステムの
各コンポーネントが正しく動作することを検証します。
"""

import torch
import sys
import traceback
from typing import Tuple, Dict, Any


def test_imports() -> bool:
    """必要なモジュールのインポートテスト"""
    try:
        print("📦 インポートテスト...")

        # 基本コンポーネント
        from adaptive_scheduler import AdaptiveInferenceScheduler
        from cache_manager import TieredCacheManager, CacheTier
        from generate_adaptive import generate_with_adaptive_scheduling

        # 既存モジュール
        from generate import generate_adaptive, compare_generation_methods

        print("✅ 全モジュールのインポート成功")
        return True

    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False


def test_scheduler_creation() -> bool:
    """スケジューラー作成テスト"""
    try:
        print("\n🔧 スケジューラー作成テスト...")

        from adaptive_scheduler import AdaptiveInferenceScheduler

        # デフォルト設定
        scheduler = AdaptiveInferenceScheduler()
        assert scheduler.min_block_size == 8
        assert scheduler.max_block_size == 64
        assert scheduler.base_confidence_threshold == 0.8

        # カスタム設定
        custom_scheduler = AdaptiveInferenceScheduler(
            min_block_size=4,
            max_block_size=32,
            base_confidence_threshold=0.9
        )
        assert custom_scheduler.min_block_size == 4
        assert custom_scheduler.max_block_size == 32
        assert custom_scheduler.base_confidence_threshold == 0.9

        print("✅ スケジューラー作成成功")
        return True

    except Exception as e:
        print(f"❌ スケジューラー作成エラー: {e}")
        traceback.print_exc()
        return False


def test_cache_manager_creation() -> bool:
    """キャッシュマネージャー作成テスト"""
    try:
        print("\n💾 キャッシュマネージャー作成テスト...")

        from cache_manager import TieredCacheManager, CacheTier

        # デフォルト設定
        cache_manager = TieredCacheManager()
        assert cache_manager.tier2_stability_threshold == 0.85
        assert cache_manager.tier2_update_interval == 3

        # 階層列挙型
        assert CacheTier.FROZEN.value == "frozen"
        assert CacheTier.STABLE.value == "stable"
        assert CacheTier.ACTIVE.value == "active"

        # 状態リセット
        cache_manager.reset_cache()
        assert cache_manager.tier1_cache is None
        assert len(cache_manager.tier2_cache) == 0

        print("✅ キャッシュマネージャー作成成功")
        return True

    except Exception as e:
        print(f"❌ キャッシュマネージャー作成エラー: {e}")
        traceback.print_exc()
        return False


def test_scheduler_adaptation() -> bool:
    """スケジューラー適応ロジックテスト"""
    try:
        print("\n🎯 スケジューラー適応ロジックテスト...")

        from adaptive_scheduler import AdaptiveInferenceScheduler

        scheduler = AdaptiveInferenceScheduler(
            min_block_size=8,
            max_block_size=32,
            base_confidence_threshold=0.8
        )

        # ダミーデータでテスト
        dummy_logits = torch.randn(1, 10, 1000)  # (batch, seq, vocab)
        dummy_tokens = torch.randint(0, 1000, (1, 10))  # (batch, seq)
        dummy_mask = torch.ones(1, 10, dtype=torch.bool)  # (batch, seq)

        # 適応ステップ実行
        next_size, adapted_threshold, metrics = scheduler.step(
            logits=dummy_logits,
            tokens=dummy_tokens,
            mask_index=dummy_mask,
            step_num=1,
            total_steps=10
        )

        # 結果検証
        assert isinstance(next_size, int)
        assert scheduler.min_block_size <= next_size <= scheduler.max_block_size
        assert isinstance(adapted_threshold, float)
        assert isinstance(metrics, dict)
        assert 'confidence' in metrics
        assert 'entropy' in metrics

        print("✅ スケジューラー適応ロジック成功")
        return True

    except Exception as e:
        print(f"❌ スケジューラー適応ロジックエラー: {e}")
        traceback.print_exc()
        return False


def test_cache_operations() -> bool:
    """キャッシュ操作テスト"""
    try:
        print("\n📚 キャッシュ操作テスト...")

        from cache_manager import TieredCacheManager

        cache_manager = TieredCacheManager()

        # ダミーキャッシュデータ
        dummy_cache = [
            (torch.randn(1, 8, 20, 64), torch.randn(1, 8, 20, 64))  # (key, value)
            for _ in range(4)  # 4層分
        ]

        # プロンプトキャッシュ設定
        cache_manager.set_prompt_cache(20, dummy_cache)
        assert cache_manager.is_prompt_cached
        assert cache_manager.prompt_length == 20

        # ブロック分類テスト
        confidence_scores = torch.tensor([0.9, 0.8, 0.85, 0.7])
        tier = cache_manager.classify_block(0, confidence_scores)
        assert tier in [cache_manager.CacheTier.STABLE,
                        cache_manager.CacheTier.ACTIVE]

        # メトリクス取得
        metrics = cache_manager.get_cache_efficiency_metrics()
        assert isinstance(metrics, dict)
        assert 'cache_hit_rate' in metrics

        print("✅ キャッシュ操作成功")
        return True

    except Exception as e:
        print(f"❌ キャッシュ操作エラー: {e}")
        traceback.print_exc()
        return False


def test_integration_imports() -> bool:
    """統合インポートテスト"""
    try:
        print("\n🔗 統合インポートテスト...")

        # 統合関数のインポート
        from generate import generate_adaptive, compare_generation_methods
        from generate import ADAPTIVE_SCHEDULING_AVAILABLE

        print(f"   アダプティブスケジューリング利用可能: {ADAPTIVE_SCHEDULING_AVAILABLE}")

        # 関数が呼び出し可能か確認
        assert callable(generate_adaptive)
        assert callable(compare_generation_methods)

        print("✅ 統合インポート成功")
        return True

    except Exception as e:
        print(f"❌ 統合インポートエラー: {e}")
        traceback.print_exc()
        return False


def run_verification() -> Tuple[bool, Dict[str, bool]]:
    """全体検証の実行"""
    print("🚀 Self-Correcting Adaptive Inference Scheduling 実装検証")
    print("=" * 70)

    tests = [
        ("インポート", test_imports),
        ("スケジューラー作成", test_scheduler_creation),
        ("キャッシュマネージャー作成", test_cache_manager_creation),
        ("スケジューラー適応ロジック", test_scheduler_adaptation),
        ("キャッシュ操作", test_cache_operations),
        ("統合インポート", test_integration_imports)
    ]

    results = {}
    all_passed = True

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
            all_passed = all_passed and passed
        except Exception as e:
            print(f"❌ {test_name}で予期しないエラー: {e}")
            results[test_name] = False
            all_passed = False

    # 結果サマリー
    print("\n" + "=" * 70)
    print("📊 検証結果サマリー")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ 成功" if passed else "❌ 失敗"
        print(f"  {test_name:<25} {status}")

    overall_status = "✅ 全テスト成功" if all_passed else "❌ 一部テスト失敗"
    print(f"\n🎯 総合結果: {overall_status}")

    if all_passed:
        print("\n🎉 Self-Correcting Adaptive Inference Schedulingの実装は正常です！")
        print("   すべてのコンポーネントが正しく動作しています。")
        print("\n📚 次のステップ:")
        print("   1. QUICK_START_ADAPTIVE_SCHEDULING.md を参照して使用開始")
        print("   2. test_adaptive_scheduling.py でベンチマーク実行")
        print("   3. examples/adaptive_scheduling_demo.py でデモ確認")
    else:
        print("\n⚠️ 一部のコンポーネントで問題が発生しています。")
        print("   上記のエラーメッセージを確認して修正してください。")

    return all_passed, results


def main():
    """メイン関数"""
    try:
        success, results = run_verification()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ 検証が中断されました")
        return 1
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
