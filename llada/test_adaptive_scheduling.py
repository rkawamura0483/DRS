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
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache
from generate_adaptive import generate_with_adaptive_scheduling, generate_with_custom_scheduler
from adaptive_scheduler import AdaptiveInferenceScheduler
from cache_manager import TieredCacheManager


class AdaptiveSchedulingTester:
    """
    Self-Correcting Adaptive Inference Scheduling テストフレームワーク

    包括的なベンチマーク、アブレーション研究、比較評価を実施
    """

    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct", device: str = "auto"):
        """
        テスターの初期化

        Args:
            model_name: 使用するモデル名
            device: 使用するデバイス
        """
        self.model_name = model_name
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔧 AdaptiveSchedulingTester初期化")
        print(f"   モデル: {model_name}")
        print(f"   デバイス: {self.device}")

        # モデルとトークナイザーの読み込み
        self.model = LLaDAModelLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        # テストケース
        self.test_cases = self._prepare_test_cases()

        print(f"✅ 初期化完了 ({len(self.test_cases)} テストケース)")

    def _prepare_test_cases(self) -> List[Dict[str, Any]]:
        """多様なテストケースを準備"""
        return [
            {
                "name": "math_reasoning",
                "prompt": "Solve this step by step: If a train travels 120 km in 2 hours, what is its average speed?",
                "category": "reasoning",
                "expected_difficulty": "medium"
            },
            {
                "name": "code_generation",
                "prompt": "Write a Python function to calculate the factorial of a number:",
                "category": "code",
                "expected_difficulty": "medium"
            },
            {
                "name": "creative_writing",
                "prompt": "Write a short story about a robot discovering emotions:",
                "category": "creative",
                "expected_difficulty": "high"
            },
            {
                "name": "simple_qa",
                "prompt": "What is the capital of France?",
                "category": "factual",
                "expected_difficulty": "low",
                "expected_mode": "HIGH_EFFICIENCY"  # 簡単な事実問題では効率モードが期待される
            },
            {
                "name": "complex_reasoning",
                "prompt": "Explain the relationship between quantum mechanics and general relativity in simple terms:",
                "category": "reasoning",
                "expected_difficulty": "high",
                "expected_mode": "HIGH_QUALITY"  # 複雑な推論では品質モードが期待される
            },
            {
                "name": "list_generation",
                "prompt": "List 10 benefits of regular exercise:",
                "category": "structured",
                "expected_difficulty": "low",
                "expected_mode": "HIGH_EFFICIENCY"  # リスト生成では効率モードが期待される
            }
        ]

    def plot_confidence_movement(self, metrics: Dict[str, Any], test_case_name: str,
                                 save_plots: bool = True, scheduler_config: Dict = None) -> None:
        """
        信頼度の動きとモード切り替えをプロット

        Args:
            metrics: 生成メトリクス
            test_case_name: テストケース名
            save_plots: プロットを保存するか
            scheduler_config: スケジューラー設定（閾値線の表示用）
        """
        if 'confidence_history' not in metrics or 'mode_history' not in metrics:
            print("❌ 信頼度履歴またはモード履歴が見つかりません")
            return

        confidence_history = metrics['confidence_history']
        mode_history = metrics['mode_history']

        if len(confidence_history) == 0:
            print("❌ 信頼度履歴が空です")
            return

        # デフォルト閾値
        quality_threshold = 0.80
        efficiency_threshold = 0.95

        # スケジューラー設定から閾値を取得
        if scheduler_config:
            quality_threshold = scheduler_config.get(
                'to_quality_threshold', 0.80)
            efficiency_threshold = scheduler_config.get(
                'to_efficiency_threshold', 0.95)

        # プロットの設定
        plt.figure(figsize=(12, 8))

        # 上段: 信頼度の変化
        plt.subplot(2, 1, 1)
        steps = range(len(confidence_history))
        plt.plot(steps, confidence_history, 'b-',
                 linewidth=2, label='Confidence')

        # 閾値線を追加
        plt.axhline(y=quality_threshold, color='r', linestyle='--',
                    alpha=0.7, label=f'Quality Threshold ({quality_threshold})')
        plt.axhline(y=efficiency_threshold, color='g', linestyle='--', alpha=0.7,
                    label=f'Efficiency Threshold ({efficiency_threshold})')

        # モード変更点をハイライト
        mode_changes = []
        for i in range(1, len(mode_history)):
            if mode_history[i] != mode_history[i-1]:
                mode_changes.append(i)
                plt.axvline(x=i, color='orange', linestyle=':', alpha=0.8)

        plt.ylabel('Confidence', fontsize=12)
        plt.title(
            f'Confidence Movement - {test_case_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # 下段: モード切り替え
        plt.subplot(2, 1, 2)

        # モードを数値に変換 (HIGH_EFFICIENCY=1, HIGH_QUALITY=0)
        mode_numeric = [1 if mode ==
                        'HIGH_EFFICIENCY' else 0 for mode in mode_history]

        # モード切り替えを色分けして表示
        for i in range(len(mode_numeric)):
            color = 'lightgreen' if mode_numeric[i] == 1 else 'lightcoral'
            label = 'High-Efficiency' if mode_numeric[i] == 1 else 'High-Quality'

            # 最初の出現時のみラベルを付ける
            if i == 0 or (i > 0 and mode_numeric[i] != mode_numeric[i-1]):
                plt.bar(i, 1, color=color, alpha=0.7, label=label if i == 0 or label not in [
                        item.get_label() for item in plt.gca().get_legend_handles_labels()[1]] else "")
            else:
                plt.bar(i, 1, color=color, alpha=0.7)

        plt.ylabel('Mode', fontsize=12)
        plt.xlabel('Generation Step', fontsize=12)
        plt.title('Mode Switching Pattern', fontsize=14, fontweight='bold')
        plt.yticks([0, 1], ['High-Quality', 'High-Efficiency'])

        # 重複ラベルを避ける
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # プロットの保存
        if save_plots:
            output_dir = Path("confidence_plots")
            output_dir.mkdir(exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = output_dir / \
                f"confidence_{test_case_name}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 信頼度プロットを保存: {filename}")

        plt.show()

        # 統計情報を表示
        self._print_confidence_stats(
            confidence_history, mode_history, mode_changes)

    def _print_confidence_stats(self, confidence_history: List[float],
                                mode_history: List[str], mode_changes: List[int]) -> None:
        """信頼度統計情報を表示"""
        print(f"\n📈 信頼度統計:")
        print(f"   平均信頼度: {np.mean(confidence_history):.3f}")
        print(f"   最小信頼度: {np.min(confidence_history):.3f}")
        print(f"   最大信頼度: {np.max(confidence_history):.3f}")
        print(f"   信頼度標準偏差: {np.std(confidence_history):.3f}")
        print(f"   モード変更回数: {len(mode_changes)}")

        # モード別時間統計
        efficiency_steps = sum(
            1 for mode in mode_history if mode == 'HIGH_EFFICIENCY')
        quality_steps = sum(
            1 for mode in mode_history if mode == 'HIGH_QUALITY')
        total_steps = len(mode_history)

        print(
            f"   効率モード時間: {efficiency_steps}/{total_steps} ({efficiency_steps/total_steps*100:.1f}%)")
        print(
            f"   品質モード時間: {quality_steps}/{total_steps} ({quality_steps/total_steps*100:.1f}%)")

    def run_comprehensive_benchmark(self,
                                    gen_length: int = 128,
                                    num_runs: int = 3,
                                    save_results: bool = True,
                                    scheduler_config: Dict = None) -> Dict[str, Any]:
        """
        包括的ベンチマークを実行

        Args:
            gen_length: 生成長
            num_runs: 実行回数（平均を取る）
            save_results: 結果を保存するか
            scheduler_config: スケジューラー設定

        Returns:
            ベンチマーク結果
        """
        print(f"\n🚀 包括的ベンチマーク開始")
        print(f"   生成長: {gen_length}")
        print(f"   実行回数: {num_runs}")

        if scheduler_config:
            print(f"   スケジューラー設定: {scheduler_config}")

        methods = {
            "adaptive_scheduling": lambda tc, gl: self._run_adaptive_scheduling(tc, gl, scheduler_config),
            "dual_cache": self._run_dual_cache,
        }

        all_results = {}

        for method_name, method_func in methods.items():
            print(f"\n📊 {method_name} 評価中...")
            method_results = {}

            for test_case in tqdm(self.test_cases, desc=f"{method_name}"):
                case_results = []

                for run in range(num_runs):
                    try:
                        result = method_func(test_case, gen_length)
                        case_results.append(result)
                    except Exception as e:
                        print(f"❌ エラー in {test_case['name']}, run {run}: {e}")
                        continue

                if case_results:
                    # 平均メトリクスを計算
                    avg_result = self._average_results(case_results)
                    method_results[test_case['name']] = avg_result

            all_results[method_name] = method_results

        # 比較分析
        comparison = self._analyze_comparison(all_results)

        if save_results:
            self._save_results(all_results, comparison, gen_length)

        return {
            'detailed_results': all_results,
            'comparison': comparison,
            'config': {
                'gen_length': gen_length,
                'num_runs': num_runs,
                'test_cases': len(self.test_cases),
                'scheduler_config': scheduler_config
            }
        }

    def run_ablation_study(self, gen_length: int = 128,
                           base_scheduler_config: Dict = None) -> Dict[str, Any]:
        """
        アブレーション研究を実行

        Args:
            gen_length: 生成長
            base_scheduler_config: ベーススケジューラー設定

        Returns:
            アブレーション結果
        """
        print(f"\n🔬 アブレーション研究開始")

        # デフォルト設定
        default_config = {
            'to_quality_threshold': 0.80,
            'to_efficiency_threshold': 0.95,
            'confidence_window_size': 2,
            'high_efficiency_params': {'block_size': 32, 'threshold': 0.75},
            'high_quality_params': {'block_size': 8, 'threshold': 0.95}
        }

        # ベース設定があればマージ
        if base_scheduler_config:
            default_config.update(base_scheduler_config)

        configurations = [
            {
                "name": "full_system",
                "mode_switching": True,
                "tiered_cache": True,
                "scheduler_config": default_config.copy()
            },
            {
                "name": "aggressive_switching",
                "mode_switching": True,
                "tiered_cache": True,
                "scheduler_config": {
                    **default_config,
                    'to_quality_threshold': 0.70,
                    'to_efficiency_threshold': 0.85,
                    'confidence_window_size': 1
                }
            },
            {
                "name": "conservative_switching",
                "mode_switching": True,
                "tiered_cache": True,
                "scheduler_config": {
                    **default_config,
                    'to_quality_threshold': 0.90,
                    'to_efficiency_threshold': 0.98,
                    'confidence_window_size': 3
                }
            },
            {
                "name": "efficiency_only",
                "mode_switching": False,
                "tiered_cache": True,
                "scheduler_config": {
                    **default_config,
                    'high_quality_params': default_config['high_efficiency_params'].copy()
                }
            },
            {
                "name": "quality_only",
                "mode_switching": False,
                "tiered_cache": True,
                "scheduler_config": {
                    **default_config,
                    'high_efficiency_params': default_config['high_quality_params'].copy()
                }
            },
            {
                "name": "no_tiered_cache",
                "mode_switching": True,
                "tiered_cache": False,
                "scheduler_config": default_config.copy()
            }
        ]

        ablation_results = {}

        for config in configurations:
            print(f"\n🧪 設定: {config['name']}")
            config_results = {}

            for test_case in tqdm(self.test_cases, desc=config['name']):
                try:
                    result = self._run_ablation_config(
                        test_case, gen_length, config)
                    config_results[test_case['name']] = result
                except Exception as e:
                    print(f"❌ エラー in {test_case['name']}: {e}")
                    continue

            ablation_results[config['name']] = config_results

        # アブレーション分析
        analysis = self._analyze_ablation(ablation_results)

        return {
            'ablation_results': ablation_results,
            'analysis': analysis,
            'configurations': configurations
        }

    def run_long_context_evaluation(self, seq_lengths: List[int] = [512, 1024, 2048],
                                    scheduler_config: Dict = None) -> Dict[str, Any]:
        """
        長文コンテキスト評価

        Args:
            seq_lengths: 評価するシーケンス長のリスト
            scheduler_config: スケジューラー設定

        Returns:
            長文コンテキスト評価結果
        """
        print(f"\n📏 長文コンテキスト評価開始")

        # 長文用テストケース
        long_context_case = {
            "name": "long_context_qa",
            "prompt": "Based on the following context, answer the question: " + "A" * 200 + " Question: What is the main topic?",
            "category": "long_context",
            "expected_difficulty": "high"
        }

        results = {}

        for seq_length in seq_lengths:
            print(f"\n🔍 シーケンス長: {seq_length}")

            # アダプティブスケジューリング
            adaptive_result = self._run_adaptive_scheduling(
                long_context_case, seq_length, scheduler_config)

            # 静的手法（比較用）
            static_result = self._run_dual_cache(long_context_case, seq_length)

            results[seq_length] = {
                'adaptive': adaptive_result,
                'static': static_result,
                'speedup': static_result['total_time'] / adaptive_result['total_time'] if adaptive_result['total_time'] > 0 else 0,
                'efficiency_gain': (static_result['nfe'] - adaptive_result['nfe']) / static_result['nfe'] if static_result['nfe'] > 0 else 0
            }

            print(
                f"   アダプティブ: {adaptive_result['total_time']:.2f}s, NFE={adaptive_result['nfe']}")
            print(
                f"   静的: {static_result['total_time']:.2f}s, NFE={static_result['nfe']}")
            print(f"   スピードアップ: {results[seq_length]['speedup']:.2f}x")

        return results

    def _run_adaptive_scheduling(self, test_case: Dict, gen_length: int,
                                 scheduler_config: Dict = None) -> Dict[str, Any]:
        """モード切り替え式アダプティブスケジューリングを実行"""
        prompt = self.tokenizer.encode(
            test_case['prompt'], return_tensors='pt').to(self.device)

        # デフォルト設定
        default_scheduler_config = {
            'to_quality_threshold': 0.80,
            'to_efficiency_threshold': 0.95,
            'confidence_window_size': 2,
            'high_efficiency_params': {'block_size': 32, 'threshold': 0.75},
            'high_quality_params': {'block_size': 8, 'threshold': 0.95}
        }

        # 設定をマージ
        if scheduler_config:
            final_config = {**default_scheduler_config, **scheduler_config}
        else:
            final_config = default_scheduler_config

        start_time = time.time()
        output, metrics = generate_with_adaptive_scheduling(
            model=self.model,
            prompt=prompt,
            gen_length=gen_length,
            enable_tiered_cache=True,
            scheduler_config=final_config,
            verbose=True  # 詳細ログを有効化
        )
        end_time = time.time()

        # 生成されたテキストをデコード
        generated_text = self.tokenizer.decode(
            output[0, prompt.shape[1]:], skip_special_tokens=True)

        return {
            'method': 'adaptive_scheduling',
            'total_time': end_time - start_time,
            'nfe': metrics['nfe'],
            'adaptations': metrics['total_adaptations'],
            'avg_block_size': metrics.get('avg_block_size', 0),
            'cache_hit_rate': metrics.get('cache_efficiency', {}).get('cache_hit_rate', 0),
            'mode_changes': metrics['total_adaptations'],  # モード変更回数
            'final_mode': metrics.get('mode_history', ['UNKNOWN'])[-1] if metrics.get('mode_history') else 'UNKNOWN',
            'generated_text': generated_text,
            'text_length': len(generated_text),
            'metrics': metrics,
            'scheduler_config': final_config
        }

    def _run_dual_cache(self, test_case: Dict, gen_length: int) -> Dict[str, Any]:
        """デュアルキャッシュを実行"""
        prompt = self.tokenizer.encode(
            test_case['prompt'], return_tensors='pt').to(self.device)

        start_time = time.time()
        output, nfe = generate_with_dual_cache(
            model=self.model,
            prompt=prompt,
            steps=128,
            gen_length=gen_length,
            block_length=32,
            threshold=0.8
        )
        end_time = time.time()

        generated_text = self.tokenizer.decode(
            output[0, prompt.shape[1]:], skip_special_tokens=True)

        return {
            'method': 'dual_cache',
            'total_time': end_time - start_time,
            'nfe': nfe,
            'adaptations': 0,
            'avg_block_size': 32,
            'cache_hit_rate': 0,
            'generated_text': generated_text,
            'text_length': len(generated_text)
        }

    def _run_ablation_config(self, test_case: Dict, gen_length: int, config: Dict) -> Dict[str, Any]:
        """特定のアブレーション設定を実行"""
        prompt = self.tokenizer.encode(
            test_case['prompt'], return_tensors='pt').to(self.device)

        # モード切り替え方式に基づく設定
        scheduler_config = config['scheduler_config'].copy()

        start_time = time.time()
        output, metrics = generate_with_adaptive_scheduling(
            model=self.model,
            prompt=prompt,
            gen_length=gen_length,
            enable_tiered_cache=config['tiered_cache'],
            scheduler_config=scheduler_config,
            verbose=True  # 詳細ログを有効化
        )
        end_time = time.time()

        generated_text = self.tokenizer.decode(
            output[0, prompt.shape[1]:], skip_special_tokens=True)

        return {
            'config': config['name'],
            'total_time': end_time - start_time,
            'nfe': metrics['nfe'],
            'adaptations': metrics['total_adaptations'],
            'mode_changes': metrics['total_adaptations'],
            'final_mode': metrics.get('mode_history', ['UNKNOWN'])[-1] if metrics.get('mode_history') else 'UNKNOWN',
            'avg_block_size': metrics.get('avg_block_size', 0),
            'generated_text': generated_text,
            'metrics': metrics
        }

    def _average_results(self, results: List[Dict]) -> Dict[str, Any]:
        """複数実行の結果を平均"""
        if not results:
            return {}

        numeric_keys = ['total_time', 'nfe', 'adaptations', 'mode_changes',
                        'avg_block_size', 'cache_hit_rate', 'text_length']
        averaged = {'method': results[0]['method']}

        for key in numeric_keys:
            values = [r.get(key, 0) for r in results if key in r]
            if values:
                averaged[key] = np.mean(values)
                averaged[f'{key}_std'] = np.std(values)

        # 最初の結果からテキストを保持
        averaged['generated_text'] = results[0].get('generated_text', '')

        return averaged

    def _analyze_comparison(self, all_results: Dict) -> Dict[str, Any]:
        """手法間の比較分析"""
        comparison = {
            'summary': {},
            'per_category': {},
            'overall_metrics': {}
        }

        # 全体的なメトリクス比較
        methods = list(all_results.keys())
        if len(methods) >= 2:
            for metric in ['total_time', 'nfe', 'adaptations', 'mode_changes']:
                comparison['overall_metrics'][metric] = {}
                for method in methods:
                    values = []
                    for case_name, case_result in all_results[method].items():
                        if metric in case_result:
                            values.append(case_result[metric])
                    if values:
                        comparison['overall_metrics'][metric][method] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }

        # カテゴリ別分析
        categories = set(case['category'] for case in self.test_cases)
        for category in categories:
            comparison['per_category'][category] = {}

            # そのカテゴリのテストケースを特定
            category_cases = [case['name']
                              for case in self.test_cases if case['category'] == category]

            for method in methods:
                category_times = []
                category_nfes = []
                for case_name in category_cases:
                    if case_name in all_results[method]:
                        result = all_results[method][case_name]
                        if 'total_time' in result:
                            category_times.append(result['total_time'])
                        if 'nfe' in result:
                            category_nfes.append(result['nfe'])

                if category_times:
                    comparison['per_category'][category][method] = {
                        'avg_time': np.mean(category_times),
                        'avg_nfe': np.mean(category_nfes) if category_nfes else 0
                    }

        return comparison

    def _analyze_ablation(self, ablation_results: Dict) -> Dict[str, Any]:
        """アブレーション結果の分析"""
        analysis = {
            'component_impact': {},
            'relative_performance': {}
        }

        # フルシステムをベースラインとして使用
        if 'full_system' in ablation_results:
            baseline = ablation_results['full_system']

            for config_name, config_results in ablation_results.items():
                if config_name == 'full_system':
                    continue

                # 平均パフォーマンスを計算
                baseline_times = [r['total_time'] for r in baseline.values()]
                config_times = [r['total_time']
                                for r in config_results.values()]

                baseline_nfes = [r['nfe'] for r in baseline.values()]
                config_nfes = [r['nfe'] for r in config_results.values()]

                if baseline_times and config_times:
                    time_impact = (
                        np.mean(config_times) - np.mean(baseline_times)) / np.mean(baseline_times) * 100
                    nfe_impact = (np.mean(config_nfes) - np.mean(baseline_nfes)) / \
                        np.mean(baseline_nfes) * 100 if baseline_nfes else 0

                    analysis['component_impact'][config_name] = {
                        'time_impact_percent': time_impact,
                        'nfe_impact_percent': nfe_impact
                    }

        return analysis

    def _save_results(self, results: Dict, comparison: Dict, gen_length: int):
        """結果をファイルに保存"""
        output_dir = Path("adaptive_scheduling_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 詳細結果
        with open(output_dir / f"detailed_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # 比較結果
        with open(output_dir / f"comparison_{timestamp}.json", 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        print(f"💾 結果を保存: {output_dir}")

    def create_performance_plots(self, results: Dict):
        """パフォーマンス比較プロットを作成"""
        # TODO: matplotlib を使用してプロットを作成
        pass

    def quick_mode_switching_test(self, test_case_name: str = "complex_reasoning",
                                  gen_length: int = 64, scheduler_config: Dict = None,
                                  plot_confidence: bool = True) -> Dict[str, Any]:
        """
        モード切り替え機能の簡単なテスト

        Args:
            test_case_name: テストするケース名
            gen_length: 生成長
            scheduler_config: スケジューラー設定
            plot_confidence: 信頼度をプロットするか

        Returns:
            テスト結果
        """
        print(f"\n🔄 モード切り替えテスト: {test_case_name}")

        # テストケースを取得
        test_case = next(
            (case for case in self.test_cases if case['name'] == test_case_name), None)
        if not test_case:
            print(f"❌ テストケース '{test_case_name}' が見つかりません")
            return {}

        # 設定でテスト実行
        result = self._run_adaptive_scheduling(
            test_case, gen_length, scheduler_config)

        print(f"✅ テスト完了:")
        print(f"   実行時間: {result['total_time']:.2f}秒")
        print(f"   NFE: {result['nfe']}")
        print(f"   モード変更: {result['mode_changes']}回")
        print(f"   最終モード: {result['final_mode']}")
        print(f"   平均ブロックサイズ: {result['avg_block_size']:.1f}")
        print(f"   期待モード: {test_case.get('expected_mode', '未定義')}")
        print(f"   使用設定: {result['scheduler_config']}")

        # 信頼度プロットを生成
        if plot_confidence and 'metrics' in result:
            print(f"\n📊 信頼度プロット生成中...")
            self.plot_confidence_movement(result['metrics'], test_case_name,
                                          save_plots=True, scheduler_config=scheduler_config)

        return result

    def compare_parameter_settings(self, test_case_name: str = "complex_reasoning",
                                   gen_length: int = 64) -> None:
        """
        異なるパラメータ設定を比較

        Args:
            test_case_name: テストするケース名
            gen_length: 生成長
        """
        print(f"\n🔍 パラメータ設定比較: {test_case_name}")

        # 異なる設定を定義
        configurations = [
            {
                'name': 'デフォルト',
                'config': {
                    'to_quality_threshold': 0.80,
                    'to_efficiency_threshold': 0.95,
                    'confidence_window_size': 2
                }
            },
            {
                'name': 'アグレッシブ',
                'config': {
                    'to_quality_threshold': 0.70,
                    'to_efficiency_threshold': 0.85,
                    'confidence_window_size': 1
                }
            },
            {
                'name': '保守的',
                'config': {
                    'to_quality_threshold': 0.90,
                    'to_efficiency_threshold': 0.98,
                    'confidence_window_size': 3
                }
            }
        ]

        results = {}

        for config_info in configurations:
            print(f"\n🧪 テスト中: {config_info['name']}")

            # ベース設定を更新
            full_config = {
                **config_info['config'],
                'high_efficiency_params': {'block_size': 32, 'threshold': 0.75},
                'high_quality_params': {'block_size': 8, 'threshold': 0.95}
            }

            result = self.quick_mode_switching_test(
                test_case_name=test_case_name,
                gen_length=gen_length,
                scheduler_config=full_config,
                plot_confidence=False  # 比較時は個別プロットを無効化
            )

            results[config_info['name']] = {
                'result': result,
                'config': full_config
            }

        # 比較結果を表示
        print(f"\n📊 設定比較結果:")
        print(f"{'設定':<12} {'時間(s)':<10} {'NFE':<8} {'モード変更':<10} {'効率モード(%)':<15}")
        print("-" * 65)

        for name, data in results.items():
            result = data['result']
            if 'metrics' in result:
                metrics = result['metrics']
                mode_history = metrics.get('mode_history', [])
                efficiency_percent = sum(1 for mode in mode_history if mode ==
                                         'HIGH_EFFICIENCY') / len(mode_history) * 100 if mode_history else 0

                print(
                    f"{name:<12} {result['total_time']:<10.2f} {result['nfe']:<8} {result['mode_changes']:<10} {efficiency_percent:<15.1f}")

        # 統合プロットを作成（すべての設定の信頼度を重ねて表示）
        self._create_comparison_plot(results, test_case_name)

    def _create_comparison_plot(self, results: Dict, test_case_name: str) -> None:
        """
        複数設定の比較プロットを作成

        Args:
            results: 比較結果
            test_case_name: テストケース名
        """
        plt.figure(figsize=(15, 10))

        colors = ['blue', 'red', 'green', 'purple', 'orange']

        # 上段: 信頼度比較
        plt.subplot(2, 1, 1)

        for i, (name, data) in enumerate(results.items()):
            result = data['result']
            config = data['config']

            if 'metrics' in result and 'confidence_history' in result['metrics']:
                confidence_history = result['metrics']['confidence_history']
                steps = range(len(confidence_history))

                plt.plot(steps, confidence_history,
                         color=colors[i % len(colors)], linewidth=2, label=f'{name}')

                # 各設定の閾値線
                quality_threshold = config.get('to_quality_threshold', 0.80)
                efficiency_threshold = config.get(
                    'to_efficiency_threshold', 0.95)

                plt.axhline(y=quality_threshold, color=colors[i % len(colors)],
                            linestyle='--', alpha=0.3, linewidth=1)
                plt.axhline(y=efficiency_threshold, color=colors[i % len(colors)],
                            linestyle=':', alpha=0.3, linewidth=1)

        plt.ylabel('Confidence', fontsize=12)
        plt.title(
            f'Confidence Comparison - {test_case_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # 下段: モード切り替えパターン比較
        plt.subplot(2, 1, 2)

        for i, (name, data) in enumerate(results.items()):
            result = data['result']

            if 'metrics' in result and 'mode_history' in result['metrics']:
                mode_history = result['metrics']['mode_history']
                mode_numeric = [
                    1 if mode == 'HIGH_EFFICIENCY' else 0 for mode in mode_history]
                steps = range(len(mode_numeric))

                # オフセットを使用して複数の設定を表示
                offset = i * 0.2
                plt.plot(steps, [m + offset for m in mode_numeric],
                         color=colors[i % len(colors)], linewidth=2, label=f'{name}',
                         marker='o', markersize=3, alpha=0.7)

        plt.ylabel('Mode + Offset', fontsize=12)
        plt.xlabel('Generation Step', fontsize=12)
        plt.title('Mode Switching Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # プロットを保存
        output_dir = Path("confidence_plots")
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"comparison_{test_case_name}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 比較プロットを保存: {filename}")

        plt.show()


def main():
    """
    使用例:
    # 基本的なベンチマーク
    python test_adaptive_scheduling.py --benchmark

    # アブレーション研究
    python test_adaptive_scheduling.py --ablation

    # クイックテスト（デフォルト設定）
    python test_adaptive_scheduling.py --quick-test complex_reasoning

    # クイックテスト（カスタム設定）
    python test_adaptive_scheduling.py --quick-test complex_reasoning --to-quality-threshold 0.75 --to-efficiency-threshold 0.90 --efficiency-block-size 16

    # パラメータ設定比較
    python test_adaptive_scheduling.py --compare-settings complex_reasoning

    # プロット無効化
    python test_adaptive_scheduling.py --quick-test simple_qa --no-plot

    # 全評価
    python test_adaptive_scheduling.py --comprehensive
    """
    parser = argparse.ArgumentParser(
        description="Adaptive Scheduling Test Suite")

    # 基本パラメータ
    parser.add_argument(
        "--model", default="GSAI-ML/LLaDA-8B-Instruct", help="Model name")
    parser.add_argument("--gen-length", type=int,
                        default=128, help="Generation length")
    parser.add_argument("--device", default="auto", help="Device to use")

    # 実行モード
    parser.add_argument("--benchmark", action="store_true",
                        help="Run comprehensive benchmark")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    parser.add_argument("--long-context", action="store_true",
                        help="Run long context evaluation")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run all evaluations")
    parser.add_argument("--quick-test", type=str, default=None,
                        help="Run quick mode switching test for specific case")
    parser.add_argument("--compare-settings", type=str, default=None,
                        help="Compare different parameter settings for specific case")

    # スケジューラー設定パラメータ
    scheduler_group = parser.add_argument_group(
        'scheduler', 'Adaptive Scheduler Configuration')
    scheduler_group.add_argument("--to-quality-threshold", type=float, default=0.80,
                                 help="Confidence threshold to switch to HIGH_QUALITY mode (default: 0.80)")
    scheduler_group.add_argument("--to-efficiency-threshold", type=float, default=0.95,
                                 help="Confidence threshold to switch to HIGH_EFFICIENCY mode (default: 0.95)")
    scheduler_group.add_argument("--confidence-window-size", type=int, default=2,
                                 help="Window size for confidence smoothing (default: 2)")

    # 効率モードパラメータ
    efficiency_group = parser.add_argument_group(
        'efficiency_mode', 'High-Efficiency Mode Parameters')
    efficiency_group.add_argument("--efficiency-block-size", type=int, default=32,
                                  help="Block size for HIGH_EFFICIENCY mode (default: 32)")
    efficiency_group.add_argument("--efficiency-threshold", type=float, default=0.75,
                                  help="Confidence threshold for HIGH_EFFICIENCY mode (default: 0.75)")

    # 品質モードパラメータ
    quality_group = parser.add_argument_group(
        'quality_mode', 'High-Quality Mode Parameters')
    quality_group.add_argument("--quality-block-size", type=int, default=8,
                               help="Block size for HIGH_QUALITY mode (default: 8)")
    quality_group.add_argument("--quality-threshold", type=float, default=0.95,
                               help="Confidence threshold for HIGH_QUALITY mode (default: 0.95)")

    # プロット設定
    plot_group = parser.add_argument_group('plotting', 'Plotting Options')
    plot_group.add_argument("--no-plot", action="store_true",
                            help="Disable confidence plotting for quick test")
    plot_group.add_argument("--save-plots", action="store_true", default=True,
                            help="Save plots to files (default: True)")

    args = parser.parse_args()

    # スケジューラー設定を構築
    scheduler_config = {
        'to_quality_threshold': args.to_quality_threshold,
        'to_efficiency_threshold': args.to_efficiency_threshold,
        'confidence_window_size': args.confidence_window_size,
        'high_efficiency_params': {
            'block_size': args.efficiency_block_size,
            'threshold': args.efficiency_threshold
        },
        'high_quality_params': {
            'block_size': args.quality_block_size,
            'threshold': args.quality_threshold
        }
    }

    # 設定の表示
    print(f"🔧 スケジューラー設定:")
    print(f"   効率→品質閾値: {args.to_quality_threshold}")
    print(f"   品質→効率閾値: {args.to_efficiency_threshold}")
    print(f"   信頼度ウィンドウ: {args.confidence_window_size}")
    print(
        f"   効率モード: ブロックサイズ={args.efficiency_block_size}, 閾値={args.efficiency_threshold}")
    print(
        f"   品質モード: ブロックサイズ={args.quality_block_size}, 閾値={args.quality_threshold}")

    # テスターの初期化
    tester = AdaptiveSchedulingTester(
        model_name=args.model, device=args.device)

    if args.quick_test:
        print("\n" + "="*50)
        print("🔄 クイックモード切り替えテスト実行")
        print("="*50)
        quick_result = tester.quick_mode_switching_test(
            test_case_name=args.quick_test,
            gen_length=args.gen_length,
            scheduler_config=scheduler_config,
            plot_confidence=not args.no_plot
        )
        print("\n✅ クイックテスト完了")
        return

    if args.compare_settings:
        print("\n" + "="*50)
        print("🔍 パラメータ設定比較実行")
        print("="*50)
        tester.compare_parameter_settings(
            test_case_name=args.compare_settings,
            gen_length=args.gen_length
        )
        print("\n✅ 設定比較完了")
        return

    if args.comprehensive or args.benchmark:
        print("\n" + "="*50)
        print("🚀 包括的ベンチマーク実行")
        print("="*50)
        benchmark_results = tester.run_comprehensive_benchmark(
            gen_length=args.gen_length,
            scheduler_config=scheduler_config)
        print("\n✅ ベンチマーク完了")

    if args.comprehensive or args.ablation:
        print("\n" + "="*50)
        print("🔬 アブレーション研究実行")
        print("="*50)
        ablation_results = tester.run_ablation_study(
            gen_length=args.gen_length,
            base_scheduler_config=scheduler_config)
        print("\n✅ アブレーション研究完了")

    if args.comprehensive or args.long_context:
        print("\n" + "="*50)
        print("📏 長文コンテキスト評価実行")
        print("="*50)
        long_context_results = tester.run_long_context_evaluation(
            scheduler_config=scheduler_config)
        print("\n✅ 長文コンテキスト評価完了")

    print("\n🎉 全評価完了!")


if __name__ == "__main__":
    main()
