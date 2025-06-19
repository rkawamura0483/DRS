#!/usr/bin/env python3
"""
Self-Correcting Adaptive Inference Scheduling Research Benchmark

このスクリプトは研究アイデアを検証するために設計されています：
GSM8K, MATH, HumanEval, MBPPで25サンプルずつ（合計100サンプル）を使用して
アダプティブスケジューリングの時間とアキュラシーを測定します。
"""

from benchmark_runner import BenchmarkRunner
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import torch
import numpy as np
from tqdm import tqdm

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent))


class ResearchBenchmark:
    """
    アダプティブスケジューリング研究用ベンチマーククラス

    研究目標:
    - アダプティブスケジューリング vs 静的手法の比較
    - 時間とアキュラシーの両方を測定
    - GSM8K, MATH, HumanEval, MBPP の4つのベンチマークで評価
    - 各25サンプル（合計100サンプル）
    """

    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct", device: str = "auto"):
        """研究ベンチマークの初期化"""
        print("🚀 Self-Correcting Adaptive Inference Scheduling Research Benchmark")
        print("="*70)

        self.runner = BenchmarkRunner(model_name=model_name, device=device)

        # 研究用設定
        self.research_config = {
            'samples_per_dataset': 25,  # 各データセット25サンプル
            'total_expected_samples': 100,  # 合計100サンプル
            'gen_length': 256,  # 生成長
            'target_datasets': ['gsm8k', 'math', 'humaneval', 'mbpp']
        }

        # アダプティブスケジューリング設定（論文の設定に基づく）
        self.adaptive_config = {
            'to_quality_threshold': 0.80,      # 品質モードへの切り替え閾値
            'to_efficiency_threshold': 0.90,   # 効率モードへの切り替え閾値
            'confidence_window_size': 2,       # 信頼度ウィンドウサイズ
            'high_efficiency_params': {        # 高効率モード設定
                'block_size': 32,
                'threshold': 0.70
            },
            'high_quality_params': {           # 高品質モード設定
                'block_size': 8,
                'threshold': 0.90
            }
        }

        print(f"📊 研究設定:")
        print(f"   各データセット: {self.research_config['samples_per_dataset']}サンプル")
        print(f"   合計予定サンプル: {self.research_config['total_expected_samples']}")
        print(
            f"   対象データセット: {', '.join(self.research_config['target_datasets'])}")
        print(f"   アダプティブ設定: {self.adaptive_config}")
        print()

    def run_research_evaluation(self) -> Dict[str, Any]:
        """
        研究評価を実行

        Returns:
            研究結果辞書
        """
        print("🔬 研究評価開始...")

        # ベンチマーク実行
        start_time = time.time()

        results = self.runner.run_comprehensive_benchmark(
            samples_per_dataset=self.research_config['samples_per_dataset'],
            gen_length=self.research_config['gen_length'],
            scheduler_config=self.adaptive_config
        )

        total_time = time.time() - start_time

        # 研究特化の分析を追加
        research_analysis = self._analyze_research_results(results)

        # 結果に研究情報を追加
        results['research_analysis'] = research_analysis
        results['total_benchmark_time'] = total_time
        results['research_config'] = self.research_config

        return results

    def _analyze_research_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        研究結果の特化分析

        Args:
            results: ベンチマーク結果

        Returns:
            研究分析結果
        """
        print("\n🔍 研究結果分析中...")

        analysis = {
            'hypothesis_validation': {},
            'performance_gains': {},
            'mode_switching_analysis': {},
            'dataset_specific_insights': {}
        }

        summary = results.get('summary', {})

        if 'adaptive' in summary and 'static' in summary:
            adaptive = summary['adaptive']
            static = summary['static']

            # 仮説検証
            analysis['hypothesis_validation'] = {
                'time_improvement_achieved': adaptive['avg_time_per_sample'] < static['avg_time_per_sample'],
                'accuracy_maintained_or_improved': adaptive['accuracy'] >= static['accuracy'],
                'nfe_reduction_achieved': adaptive['avg_nfe_per_sample'] < static['avg_nfe_per_sample'],
                'adaptation_occurred': adaptive.get('total_adaptations', 0) > 0
            }

            # パフォーマンス改善
            time_improvement = (
                static['avg_time_per_sample'] - adaptive['avg_time_per_sample']) / static['avg_time_per_sample'] * 100
            accuracy_change = (adaptive['accuracy'] - static['accuracy']) * 100
            nfe_improvement = (
                static['avg_nfe_per_sample'] - adaptive['avg_nfe_per_sample']) / static['avg_nfe_per_sample'] * 100

            analysis['performance_gains'] = {
                'time_improvement_percent': time_improvement,
                'accuracy_change_percent': accuracy_change,
                'nfe_improvement_percent': nfe_improvement,
                'total_adaptations': adaptive.get('total_adaptations', 0),
                'avg_block_size': adaptive.get('avg_block_size', 0)
            }

            # モード切り替え分析
            analysis['mode_switching_analysis'] = {
                'adaptations_per_sample': adaptive.get('total_adaptations', 0) / adaptive.get('total_samples', 1),
                # 32は静的のブロックサイズ
                'avg_block_size_vs_static': adaptive.get('avg_block_size', 0) / 32.0
            }

        # データセット別洞察
        breakdown = results.get('dataset_breakdown', {})
        for dataset_name, dataset_results in breakdown.items():
            if 'adaptive' in dataset_results and 'static' in dataset_results:
                adaptive_ds = dataset_results['adaptive']
                static_ds = dataset_results['static']

                time_improvement_ds = (
                    static_ds['avg_time'] - adaptive_ds['avg_time']) / static_ds['avg_time'] * 100
                accuracy_change_ds = (
                    adaptive_ds['accuracy'] - static_ds['accuracy']) * 100

                analysis['dataset_specific_insights'][dataset_name] = {
                    'time_improvement_percent': time_improvement_ds,
                    'accuracy_change_percent': accuracy_change_ds,
                    'adaptive_better_time': adaptive_ds['avg_time'] < static_ds['avg_time'],
                    'adaptive_better_accuracy': adaptive_ds['accuracy'] >= static_ds['accuracy']
                }

        return analysis

    def print_research_summary(self, results: Dict[str, Any]):
        """
        研究結果サマリーを表示

        Args:
            results: 研究結果
        """
        print(f"\n{'='*70}")
        print(f"📊 研究結果サマリー: Self-Correcting Adaptive Inference Scheduling")
        print(f"{'='*70}")

        # 基本結果表示
        self.runner.print_results_summary(results)

        # 研究特化分析
        analysis = results.get('research_analysis', {})

        print(f"\n🔬 研究仮説検証:")
        hypothesis = analysis.get('hypothesis_validation', {})
        print(
            f"   ✓ 時間改善達成: {'✅' if hypothesis.get('time_improvement_achieved', False) else '❌'}")
        print(
            f"   ✓ アキュラシー維持/改善: {'✅' if hypothesis.get('accuracy_maintained_or_improved', False) else '❌'}")
        print(
            f"   ✓ NFE削減達成: {'✅' if hypothesis.get('nfe_reduction_achieved', False) else '❌'}")
        print(
            f"   ✓ アダプテーション発生: {'✅' if hypothesis.get('adaptation_occurred', False) else '❌'}")

        # パフォーマンス改善
        gains = analysis.get('performance_gains', {})
        print(f"\n📈 パフォーマンス改善:")
        print(f"   時間改善: {gains.get('time_improvement_percent', 0):+.1f}%")
        print(f"   アキュラシー変化: {gains.get('accuracy_change_percent', 0):+.1f}%")
        print(f"   NFE改善: {gains.get('nfe_improvement_percent', 0):+.1f}%")
        print(f"   合計アダプテーション: {gains.get('total_adaptations', 0)}")

        # データセット別洞察
        insights = analysis.get('dataset_specific_insights', {})
        print(f"\n📚 データセット別パフォーマンス:")
        for dataset, insight in insights.items():
            print(f"   {dataset.upper()}:")
            print(
                f"     時間改善: {insight.get('time_improvement_percent', 0):+.1f}%")
            print(
                f"     アキュラシー変化: {insight.get('accuracy_change_percent', 0):+.1f}%")

        # 全体評価
        print(f"\n🏆 全体評価:")
        overall_success = (
            hypothesis.get('time_improvement_achieved', False) and
            hypothesis.get('accuracy_maintained_or_improved', False) and
            hypothesis.get('adaptation_occurred', False)
        )
        print(f"   研究仮説の成功: {'✅ 成功' if overall_success else '❌ 部分的成功'}")

        total_time = results.get('total_benchmark_time', 0)
        total_samples = results.get('config', {}).get('total_samples', 0)
        print(f"   ベンチマーク実行時間: {total_time:.1f}秒")
        print(f"   評価サンプル数: {total_samples}")

    def save_research_results(self, results: Dict[str, Any], output_dir: str = "research_results"):
        """
        研究結果を保存

        Args:
            results: 研究結果
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 詳細結果保存
        detailed_file = output_path / \
            f"adaptive_scheduling_research_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # 研究サマリー保存
        summary_file = output_path / f"research_summary_{timestamp}.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(
                "# Self-Correcting Adaptive Inference Scheduling Research Results\n\n")

            analysis = results.get('research_analysis', {})
            hypothesis = analysis.get('hypothesis_validation', {})
            gains = analysis.get('performance_gains', {})

            f.write("## Hypothesis Validation\n\n")
            f.write(
                f"- Time Improvement: {'✅' if hypothesis.get('time_improvement_achieved', False) else '❌'}\n")
            f.write(
                f"- Accuracy Maintained/Improved: {'✅' if hypothesis.get('accuracy_maintained_or_improved', False) else '❌'}\n")
            f.write(
                f"- NFE Reduction: {'✅' if hypothesis.get('nfe_reduction_achieved', False) else '❌'}\n")
            f.write(
                f"- Adaptations Occurred: {'✅' if hypothesis.get('adaptation_occurred', False) else '❌'}\n\n")

            f.write("## Performance Gains\n\n")
            f.write(
                f"- Time Improvement: {gains.get('time_improvement_percent', 0):+.1f}%\n")
            f.write(
                f"- Accuracy Change: {gains.get('accuracy_change_percent', 0):+.1f}%\n")
            f.write(
                f"- NFE Improvement: {gains.get('nfe_improvement_percent', 0):+.1f}%\n")
            f.write(
                f"- Total Adaptations: {gains.get('total_adaptations', 0)}\n\n")

            # データセット別結果
            f.write("## Dataset-Specific Results\n\n")
            breakdown = results.get('dataset_breakdown', {})
            for dataset, ds_results in breakdown.items():
                if 'adaptive' in ds_results and 'static' in ds_results:
                    adaptive = ds_results['adaptive']
                    static = ds_results['static']

                    f.write(f"### {dataset.upper()}\n")
                    f.write(
                        f"- Adaptive: Accuracy={adaptive['accuracy']*100:.1f}%, Time={adaptive['avg_time']:.2f}s\n")
                    f.write(
                        f"- Static: Accuracy={static['accuracy']*100:.1f}%, Time={static['avg_time']:.2f}s\n\n")

        print(f"\n💾 研究結果保存完了:")
        print(f"   詳細結果: {detailed_file}")
        print(f"   研究サマリー: {summary_file}")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="Self-Correcting Adaptive Inference Scheduling Research Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な研究ベンチマーク実行
  python run_research_benchmark.py
  
  # 小規模テスト（各5サンプル）
  python run_research_benchmark.py --quick-test
  
  # カスタム設定
  python run_research_benchmark.py --samples 10 --gen-length 128
        """
    )

    # 基本パラメータ
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct",
                        help="使用するモデル名")
    parser.add_argument("--device", default="auto",
                        help="使用デバイス (auto/cuda/cpu)")
    parser.add_argument("--samples", type=int, default=25,
                        help="各データセットのサンプル数 (デフォルト: 25)")
    parser.add_argument("--gen-length", type=int, default=256,
                        help="生成長 (デフォルト: 256)")

    # クイックテストオプション
    parser.add_argument("--quick-test", action="store_true",
                        help="クイックテスト (各5サンプル)")

    # 出力設定
    parser.add_argument("--output-dir", default="research_results",
                        help="結果出力ディレクトリ")

    # スケジューラー設定の微調整オプション
    scheduler_group = parser.add_argument_group('advanced', 'アドバンスド設定')
    scheduler_group.add_argument("--to-quality-threshold", type=float, default=0.80,
                                 help="品質モードへの切り替え閾値")
    scheduler_group.add_argument("--to-efficiency-threshold", type=float, default=0.90,
                                 help="効率モードへの切り替え閾値")

    args = parser.parse_args()

    # クイックテストの場合は設定を調整
    if args.quick_test:
        args.samples = 5
        args.gen_length = 128
        print("🚀 クイックテストモード: 各データセット5サンプル")

    # 研究ベンチマーク初期化
    research_benchmark = ResearchBenchmark(
        model_name=args.model,
        device=args.device
    )

    # サンプル数を調整
    research_benchmark.research_config['samples_per_dataset'] = args.samples
    research_benchmark.research_config['gen_length'] = args.gen_length

    # スケジューラー設定の微調整
    research_benchmark.adaptive_config['to_quality_threshold'] = args.to_quality_threshold
    research_benchmark.adaptive_config['to_efficiency_threshold'] = args.to_efficiency_threshold

    try:
        print(f"\n🎯 研究目標: Self-Correcting Adaptive Inference Schedulingの有効性検証")
        print(f"📊 評価規模: {args.samples * 4}サンプル ({args.samples}×4データセット)")

        # 研究評価実行
        results = research_benchmark.run_research_evaluation()

        # 結果表示
        research_benchmark.print_research_summary(results)

        # 結果保存
        research_benchmark.save_research_results(results, args.output_dir)

        print(f"\n🎉 研究ベンチマーク完了!")

        # 成功判定
        analysis = results.get('research_analysis', {})
        hypothesis = analysis.get('hypothesis_validation', {})

        if (hypothesis.get('time_improvement_achieved', False) and
                hypothesis.get('accuracy_maintained_or_improved', False)):
            print(f"✅ 研究仮説が検証されました！")
        else:
            print(f"⚠️  研究仮説の部分的検証 - さらなる分析が必要です")

    except Exception as e:
        print(f"❌ 研究ベンチマーク実行中にエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
