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
                "expected_difficulty": "low"
            },
            {
                "name": "complex_reasoning",
                "prompt": "Explain the relationship between quantum mechanics and general relativity in simple terms:",
                "category": "reasoning",
                "expected_difficulty": "high"
            },
            {
                "name": "list_generation",
                "prompt": "List 10 benefits of regular exercise:",
                "category": "structured",
                "expected_difficulty": "low"
            }
        ]

    def run_comprehensive_benchmark(self,
                                    gen_length: int = 128,
                                    num_runs: int = 3,
                                    save_results: bool = True) -> Dict[str, Any]:
        """
        包括的ベンチマークを実行

        Args:
            gen_length: 生成長
            num_runs: 実行回数（平均を取る）
            save_results: 結果を保存するか

        Returns:
            ベンチマーク結果
        """
        print(f"\n🚀 包括的ベンチマーク開始")
        print(f"   生成長: {gen_length}")
        print(f"   実行回数: {num_runs}")

        methods = {
            "adaptive_scheduling": self._run_adaptive_scheduling,
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
                'test_cases': len(self.test_cases)
            }
        }

    def run_ablation_study(self, gen_length: int = 128) -> Dict[str, Any]:
        """
        アブレーション研究を実行

        Args:
            gen_length: 生成長

        Returns:
            アブレーション結果
        """
        print(f"\n🔬 アブレーション研究開始")

        configurations = [
            {
                "name": "full_system",
                "dynamic_block": True,
                "adaptive_threshold": True,
                "tiered_cache": True
            },
            {
                "name": "no_dynamic_block",
                "dynamic_block": False,
                "adaptive_threshold": True,
                "tiered_cache": True
            },
            {
                "name": "no_adaptive_threshold",
                "dynamic_block": True,
                "adaptive_threshold": False,
                "tiered_cache": True
            },
            {
                "name": "no_tiered_cache",
                "dynamic_block": True,
                "adaptive_threshold": True,
                "tiered_cache": False
            },
            {
                "name": "minimal_system",
                "dynamic_block": False,
                "adaptive_threshold": False,
                "tiered_cache": False
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

    def run_long_context_evaluation(self, seq_lengths: List[int] = [512, 1024, 2048]) -> Dict[str, Any]:
        """
        長文コンテキスト評価

        Args:
            seq_lengths: 評価するシーケンス長のリスト

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
                long_context_case, seq_length)

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

    def _run_adaptive_scheduling(self, test_case: Dict, gen_length: int) -> Dict[str, Any]:
        """アダプティブスケジューリングを実行"""
        prompt = self.tokenizer.encode(
            test_case['prompt'], return_tensors='pt').to(self.device)

        start_time = time.time()
        output, metrics = generate_with_adaptive_scheduling(
            model=self.model,
            prompt=prompt,
            gen_length=gen_length,
            base_block_size=16,
            base_confidence_threshold=0.8,
            adaptation_rate=0.2,
            enable_tiered_cache=True,
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
            'generated_text': generated_text,
            'text_length': len(generated_text),
            'metrics': metrics
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

        # 設定に基づいてスケジューラーを構築
        if config['dynamic_block']:
            scheduler_config = {}
        else:
            scheduler_config = {
                'min_block_size': 16,
                'max_block_size': 16,  # 固定サイズ
                'scale_up_factor': 1.0,
                'scale_down_factor': 1.0
            }

        if not config['adaptive_threshold']:
            scheduler_config.update({
                'min_threshold': 0.8,
                'max_threshold': 0.8,  # 固定閾値
                'safety_factor': 1.0,
                'efficiency_factor': 1.0
            })

        start_time = time.time()
        output, metrics = generate_with_adaptive_scheduling(
            model=self.model,
            prompt=prompt,
            gen_length=gen_length,
            base_block_size=16,
            base_confidence_threshold=0.8,
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
            'generated_text': generated_text,
            'metrics': metrics
        }

    def _average_results(self, results: List[Dict]) -> Dict[str, Any]:
        """複数実行の結果を平均"""
        if not results:
            return {}

        numeric_keys = ['total_time', 'nfe', 'adaptations',
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
            for metric in ['total_time', 'nfe', 'adaptations']:
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


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Scheduling Test Suite")
    parser.add_argument(
        "--model", default="GSAI-ML/LLaDA-8B-Instruct", help="Model name")
    parser.add_argument("--gen-length", type=int,
                        default=128, help="Generation length")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run comprehensive benchmark")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    parser.add_argument("--long-context", action="store_true",
                        help="Run long context evaluation")
    parser.add_argument("--comprehensive", action="store_true",
                        help="Run all evaluations")
    parser.add_argument("--device", default="auto", help="Device to use")

    args = parser.parse_args()

    # テスターの初期化
    tester = AdaptiveSchedulingTester(
        model_name=args.model, device=args.device)

    if args.comprehensive or args.benchmark:
        print("\n" + "="*50)
        print("🚀 包括的ベンチマーク実行")
        print("="*50)
        benchmark_results = tester.run_comprehensive_benchmark(
            gen_length=args.gen_length)
        print("\n✅ ベンチマーク完了")

    if args.comprehensive or args.ablation:
        print("\n" + "="*50)
        print("🔬 アブレーション研究実行")
        print("="*50)
        ablation_results = tester.run_ablation_study(
            gen_length=args.gen_length)
        print("\n✅ アブレーション研究完了")

    if args.comprehensive or args.long_context:
        print("\n" + "="*50)
        print("📏 長文コンテキスト評価実行")
        print("="*50)
        long_context_results = tester.run_long_context_evaluation()
        print("\n✅ 長文コンテキスト評価完了")

    print("\n🎉 全評価完了!")


if __name__ == "__main__":
    main()
