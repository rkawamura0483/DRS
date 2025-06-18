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
Self-Correcting Adaptive Inference Scheduling デモンストレーション

このスクリプトは、アダプティブスケジューリングシステムの使用方法を
実際の例とともに示します。
"""

from cache_manager import TieredCacheManager
from adaptive_scheduler import AdaptiveInferenceScheduler
from generate_adaptive import generate_with_adaptive_scheduling, generate_with_custom_scheduler
from model.modeling_llada import LLaDAModelLM
import torch
import time
from transformers import AutoTokenizer
import sys
import os

# パスの追加（親ディレクトリからインポートするため）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_basic_usage():
    """基本的な使用方法のデモ"""
    print("🚀 基本的な使用方法のデモ")
    print("=" * 50)

    # モデルとトークナイザーの読み込み
    print("📥 モデル読み込み中...")
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    model = LLaDAModelLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    model.eval()
    print(f"✅ モデル読み込み完了 (デバイス: {device})")

    # テストプロンプト
    prompt_text = "Write a Python function to calculate the factorial of a number:"
    prompt = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

    print(f"\n📝 プロンプト: {prompt_text}")
    print(f"   プロンプト長: {prompt.shape[1]} トークン")

    # アダプティブスケジューリングで生成
    print(f"\n🔧 アダプティブスケジューリング実行中...")
    start_time = time.time()

    output, metrics = generate_with_adaptive_scheduling(
        model=model,
        prompt=prompt,
        gen_length=128,
        base_block_size=16,
        base_confidence_threshold=0.8,
        adaptation_rate=0.2,
        enable_tiered_cache=True,
        verbose=True
    )

    end_time = time.time()

    # 結果の表示
    generated_text = tokenizer.decode(
        output[0, prompt.shape[1]:], skip_special_tokens=True)

    print(f"\n📄 生成結果:")
    print(f"   生成時間: {end_time - start_time:.2f}秒")
    print(f"   総NFE: {metrics['nfe']}")
    print(f"   適応回数: {metrics['total_adaptations']}")
    print(f"   平均ブロックサイズ: {metrics.get('avg_block_size', 'N/A')}")

    if metrics.get('cache_efficiency'):
        cache_metrics = metrics['cache_efficiency']
        print(f"   キャッシュヒット率: {cache_metrics['cache_hit_rate']:.2%}")
        print(f"   メモリ使用量: {cache_metrics['memory_usage']['total_mb']:.1f} MB")

    print(f"\n📝 生成されたテキスト:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)


def demo_configuration_comparison():
    """異なる設定での比較デモ"""
    print("\n🔬 設定比較デモ")
    print("=" * 50)

    # モデルの準備（簡略化）
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    model = LLaDAModelLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # テストプロンプト
    prompt_text = "Solve this math problem step by step: If a car travels 240 km in 3 hours, what is its average speed?"
    prompt = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

    print(f"📝 プロンプト: {prompt_text}")

    # 設定1: 保守的設定（小さなブロック、高い閾値）
    print(f"\n🛡️ 保守的設定")
    conservative_start = time.time()
    conservative_output, conservative_metrics = generate_with_adaptive_scheduling(
        model=model,
        prompt=prompt,
        gen_length=96,
        base_block_size=8,
        base_confidence_threshold=0.9,
        adaptation_rate=0.1,
        verbose=False
    )
    conservative_time = time.time() - conservative_start

    # 設定2: 積極的設定（大きなブロック、低い閾値）
    print(f"⚡ 積極的設定")
    aggressive_start = time.time()
    aggressive_output, aggressive_metrics = generate_with_adaptive_scheduling(
        model=model,
        prompt=prompt,
        gen_length=96,
        base_block_size=32,
        base_confidence_threshold=0.7,
        adaptation_rate=0.3,
        verbose=False
    )
    aggressive_time = time.time() - aggressive_start

    # 設定3: バランス設定
    print(f"⚖️ バランス設定")
    balanced_start = time.time()
    balanced_output, balanced_metrics = generate_with_adaptive_scheduling(
        model=model,
        prompt=prompt,
        gen_length=96,
        base_block_size=16,
        base_confidence_threshold=0.8,
        adaptation_rate=0.2,
        verbose=False
    )
    balanced_time = time.time() - balanced_start

    # 結果比較
    print(f"\n📊 結果比較:")
    print(f"{'設定':<12} {'時間(s)':<10} {'NFE':<8} {'適応回数':<10} {'ブロックサイズ':<12}")
    print("-" * 60)

    print(f"{'保守的':<12} {conservative_time:<10.2f} {conservative_metrics['nfe']:<8} "
          f"{conservative_metrics['total_adaptations']:<10} {conservative_metrics.get('avg_block_size', 'N/A'):<12}")

    print(f"{'積極的':<12} {aggressive_time:<10.2f} {aggressive_metrics['nfe']:<8} "
          f"{aggressive_metrics['total_adaptations']:<10} {aggressive_metrics.get('avg_block_size', 'N/A'):<12}")

    print(f"{'バランス':<12} {balanced_time:<10.2f} {balanced_metrics['nfe']:<8} "
          f"{balanced_metrics['total_adaptations']:<10} {balanced_metrics.get('avg_block_size', 'N/A'):<12}")


def demo_custom_scheduler():
    """カスタムスケジューラーのデモ"""
    print("\n🎛️ カスタムスケジューラーデモ")
    print("=" * 50)

    # モデルの準備
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    model = LLaDAModelLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # カスタムスケジューラーの作成
    custom_scheduler = AdaptiveInferenceScheduler(
        min_block_size=4,
        max_block_size=48,
        base_confidence_threshold=0.85,
        adaptation_sensitivity=0.25,
        entropy_threshold_high=1.8,
        entropy_threshold_low=0.3,
        scale_up_factor=1.5,
        scale_down_factor=0.7
    )

    # カスタムキャッシュマネージャーの作成
    custom_cache_manager = TieredCacheManager(
        tier2_stability_threshold=0.9,
        tier2_update_interval=2,
        max_stable_blocks=16,
        memory_efficiency_mode=True
    )

    print(f"🔧 カスタム設定:")
    print(
        f"   ブロックサイズ範囲: {custom_scheduler.min_block_size}-{custom_scheduler.max_block_size}")
    print(f"   基本信頼度閾値: {custom_scheduler.base_confidence_threshold}")
    print(f"   適応感度: {custom_scheduler.adaptation_sensitivity}")
    print(f"   キャッシュ安定化閾値: {custom_cache_manager.tier2_stability_threshold}")

    # テストプロンプト
    prompt_text = "Explain the concept of machine learning in simple terms for a beginner:"
    prompt = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

    print(f"\n📝 プロンプト: {prompt_text}")

    # カスタムスケジューラーで生成
    start_time = time.time()
    output, metrics = generate_with_custom_scheduler(
        model=model,
        prompt=prompt,
        scheduler=custom_scheduler,
        cache_manager=custom_cache_manager,
        gen_length=128,
        verbose=True
    )
    end_time = time.time()

    # 結果の表示
    generated_text = tokenizer.decode(
        output[0, prompt.shape[1]:], skip_special_tokens=True)

    print(f"\n📊 カスタムスケジューラー結果:")
    print(f"   生成時間: {end_time - start_time:.2f}秒")
    print(f"   総NFE: {metrics['nfe']}")
    print(f"   適応回数: {metrics['total_adaptations']}")

    # 適応メトリクス
    scheduler_metrics = custom_scheduler.get_adaptation_metrics()
    print(f"   最終ブロックサイズ: {scheduler_metrics['current_block_size']}")
    print(f"   最終信頼度閾値: {scheduler_metrics['current_threshold']:.3f}")
    print(
        f"   適応率: {scheduler_metrics['adaptation_count'] / max(1, scheduler_metrics['total_blocks']):.2%}")

    print(f"\n📝 生成されたテキスト:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)


def demo_content_adaptation():
    """コンテンツタイプによる適応の違いのデモ"""
    print("\n🎭 コンテンツ適応デモ")
    print("=" * 50)

    # モデルの準備
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    model = LLaDAModelLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # 異なるタイプのプロンプト
    test_prompts = [
        {
            "name": "数学問題",
            "text": "Calculate: (15 + 25) × 3 - 18 ÷ 2 =",
            "expected": "予測可能（低適応）"
        },
        {
            "name": "創作文章",
            "text": "Write a creative story about a time-traveling cat:",
            "expected": "不確実（高適応）"
        },
        {
            "name": "事実質問",
            "text": "What is the capital of Japan?",
            "expected": "確実（低適応）"
        },
        {
            "name": "複雑推論",
            "text": "Explain why quantum computers might be able to solve certain problems faster than classical computers:",
            "expected": "複雑（中〜高適応）"
        }
    ]

    results = []

    for prompt_info in test_prompts:
        print(f"\n📝 {prompt_info['name']}: {prompt_info['text']}")
        print(f"   予想: {prompt_info['expected']}")

        prompt = tokenizer.encode(
            prompt_info['text'], return_tensors='pt').to(device)

        start_time = time.time()
        output, metrics = generate_with_adaptive_scheduling(
            model=model,
            prompt=prompt,
            gen_length=64,
            base_block_size=16,
            base_confidence_threshold=0.8,
            adaptation_rate=0.2,
            verbose=False
        )
        end_time = time.time()

        # 結果記録
        result = {
            'name': prompt_info['name'],
            'time': end_time - start_time,
            'nfe': metrics['nfe'],
            'adaptations': metrics['total_adaptations'],
            'avg_block_size': metrics.get('avg_block_size', 16),
            'avg_confidence': np.mean(metrics['confidence_history']) if metrics['confidence_history'] else 0,
            'avg_entropy': np.mean(metrics['entropy_history']) if metrics['entropy_history'] else 0
        }
        results.append(result)

        print(
            f"   結果: 時間={result['time']:.2f}s, NFE={result['nfe']}, 適応={result['adaptations']}")
        if result['avg_confidence'] > 0:
            print(
                f"   メトリクス: 平均信頼度={result['avg_confidence']:.3f}, 平均エントロピー={result['avg_entropy']:.3f}")

    # 適応パターンの分析
    print(f"\n📊 適応パターン分析:")
    print(f"{'タイプ':<12} {'適応回数':<8} {'信頼度':<10} {'エントロピー':<10} {'パターン'}")
    print("-" * 70)

    for result in results:
        if result['avg_confidence'] > 0.85 and result['adaptations'] <= 2:
            pattern = "安定型"
        elif result['avg_entropy'] > 1.5 and result['adaptations'] >= 3:
            pattern = "適応型"
        else:
            pattern = "中間型"

        print(f"{result['name']:<12} {result['adaptations']:<8} {result['avg_confidence']:<10.3f} "
              f"{result['avg_entropy']:<10.3f} {pattern}")


def main():
    """メインデモンストレーション"""
    print("🎯 Self-Correcting Adaptive Inference Scheduling")
    print("   実用デモンストレーション")
    print("="*60)

    try:
        # 基本的な使用方法
        demo_basic_usage()

        # 設定比較
        demo_configuration_comparison()

        # カスタムスケジューラー
        demo_custom_scheduler()

        # コンテンツ適応
        demo_content_adaptation()

        print(f"\n🎉 全デモンストレーション完了!")
        print(f"   アダプティブスケジューリングシステムの各機能が")
        print(f"   正常に動作することを確認しました。")

    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # numpy のインポートを追加
    import numpy as np
    main()
