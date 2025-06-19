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

import torch
import numpy as np
import time
import json
import re
import sys
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import argparse

from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache
from generate_adaptive import generate_with_adaptive_scheduling
from datasets import load_dataset
import signal


class TimeoutError(Exception):
    """カスタムタイムアウトエラー"""
    pass


def timeout_handler(signum, frame):
    """タイムアウトハンドラー"""
    raise TimeoutError("コード実行がタイムアウトしました")


class BenchmarkRunner:
    """
    研究用ベンチマークランナー

    GSM8K, MATH, HumanEval, MBPPでアダプティブスケジューリングのパフォーマンスを評価
    """

    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct", device: str = "auto"):
        """
        ベンチマークランナーの初期化

        Args:
            model_name: 使用するモデル名
            device: 使用するデバイス
        """
        self.model_name = model_name
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔧 BenchmarkRunner初期化")
        print(f"   モデル: {model_name}")
        print(f"   デバイス: {self.device}")

        # モデルとトークナイザーの読み込み
        print("📦 モデル読み込み中...")
        self.model = LLaDAModelLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

        print("✅ 初期化完了")

    def load_datasets(self, samples_per_dataset: int = 25) -> Dict[str, List[Dict]]:
        """
        ベンチマークデータセットを読み込み

        Args:
            samples_per_dataset: 各データセットから取得するサンプル数

        Returns:
            データセット辞書
        """
        print(f"📚 データセット読み込み中... (各{samples_per_dataset}サンプル)")

        datasets = {}

        # GSM8K
        print("   GSM8K読み込み中...")
        try:
            gsm8k = load_dataset("gsm8k", "main", split="test")
            datasets["gsm8k"] = [
                {
                    "dataset": "gsm8k",
                    "id": i,
                    "question": item["question"],
                    "answer": item["answer"],
                    "type": "math"
                }
                for i, item in enumerate(gsm8k.select(range(min(samples_per_dataset, len(gsm8k)))))
            ]
            print(f"     ✅ GSM8K: {len(datasets['gsm8k'])}サンプル")
        except Exception as e:
            print(f"     ❌ GSM8K読み込みエラー: {e}")
            datasets["gsm8k"] = []

        # MATH データセット
        print("   MATH読み込み中...")
        try:
            math_dataset = load_dataset("competition_math", split="test")
            datasets["math"] = [
                {
                    "dataset": "math",
                    "id": i,
                    "question": item["problem"],
                    "answer": item["solution"],
                    "type": "math"
                }
                for i, item in enumerate(math_dataset.select(range(min(samples_per_dataset, len(math_dataset)))))
            ]
            print(f"     ✅ MATH: {len(datasets['math'])}サンプル")
        except Exception as e:
            print(f"     ❌ MATH読み込みエラー: {e}")
            datasets["math"] = []

        # HumanEval
        print("   HumanEval読み込み中...")
        try:
            humaneval = load_dataset("openai_humaneval", split="test")
            datasets["humaneval"] = [
                {
                    "dataset": "humaneval",
                    "id": i,
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                    "type": "code"
                }
                for i, item in enumerate(humaneval.select(range(min(samples_per_dataset, len(humaneval)))))
            ]
            print(f"     ✅ HumanEval: {len(datasets['humaneval'])}サンプル")
        except Exception as e:
            print(f"     ❌ HumanEval読み込みエラー: {e}")
            datasets["humaneval"] = []

        # MBPP
        print("   MBPP読み込み中...")
        try:
            mbpp = load_dataset("mbpp", "sanitized", split="test")
            datasets["mbpp"] = [
                {
                    "dataset": "mbpp",
                    "id": i,
                    "task_id": item["task_id"],
                    "text": item["text"],
                    "code": item["code"],
                    "test_list": item["test_list"],
                    "type": "code"
                }
                for i, item in enumerate(mbpp.select(range(min(samples_per_dataset, len(mbpp)))))
            ]
            print(f"     ✅ MBPP: {len(datasets['mbpp'])}サンプル")
        except Exception as e:
            print(f"     ❌ MBPP読み込みエラー: {e}")
            datasets["mbpp"] = []

        total_samples = sum(len(dataset) for dataset in datasets.values())
        print(f"📊 合計: {total_samples}サンプル読み込み完了")

        return datasets

    def format_prompt(self, sample: Dict) -> str:
        """
        サンプルに応じてプロンプトをフォーマット

        Args:
            sample: データサンプル

        Returns:
            フォーマット済みプロンプト
        """
        if sample["dataset"] == "gsm8k":
            return f"問題を段階的に解いてください:\n\n{sample['question']}\n\n答え:"

        elif sample["dataset"] == "math":
            return f"以下の数学問題を解いてください:\n\n{sample['question']}\n\n解答:"

        elif sample["dataset"] == "humaneval":
            return f"{sample['prompt']}"

        elif sample["dataset"] == "mbpp":
            return f"以下の問題に対するPython関数を書いてください:\n\n{sample['text']}\n\n```python\n"

        else:
            return sample.get("question", sample.get("prompt", ""))

    def extract_answer(self, generated_text: str, sample: Dict) -> str:
        """
        生成されたテキストから答えを抽出

        Args:
            generated_text: 生成されたテキスト
            sample: 元のサンプル

        Returns:
            抽出された答え
        """
        if sample["type"] == "math":
            # 数学問題: 数値を抽出
            # 最後の数値を答えとして取得
            numbers = re.findall(r'-?\d+\.?\d*', generated_text)
            return numbers[-1] if numbers else ""

        elif sample["type"] == "code":
            # コード問題: 関数定義を抽出
            if sample["dataset"] == "humaneval":
                # def で始まる行から次のdef または文末まで
                lines = generated_text.split('\n')
                code_lines = []
                in_function = False

                for line in lines:
                    if line.strip().startswith('def '):
                        in_function = True
                        code_lines.append(line)
                    elif in_function:
                        if line.strip().startswith('def ') and len(code_lines) > 1:
                            break
                        code_lines.append(line)
                        if line.strip() == '' and len(code_lines) > 5:
                            # 空行で関数終了の可能性
                            break

                return '\n'.join(code_lines)

            elif sample["dataset"] == "mbpp":
                # ```python と ``` の間、または最初の関数定義
                code_match = re.search(
                    r'```python\n(.*?)\n```', generated_text, re.DOTALL)
                if code_match:
                    return code_match.group(1)

                # def で始まる行から推測
                lines = generated_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip().startswith('def '):
                        return '\n'.join(lines[i:i+10])  # 10行まで取得

                return generated_text[:200]  # フォールバック

        return generated_text.strip()

    def evaluate_math_answer(self, generated_answer: str, correct_answer: str) -> bool:
        """
        数学問題の答えを評価

        Args:
            generated_answer: 生成された答え
            correct_answer: 正解

        Returns:
            正解かどうか
        """
        try:
            # 数値抽出
            gen_nums = re.findall(r'-?\d+\.?\d*', generated_answer)
            correct_nums = re.findall(r'-?\d+\.?\d*', correct_answer)

            if not gen_nums or not correct_nums:
                return False

            # 最後の数値を比較
            gen_val = float(gen_nums[-1])
            correct_val = float(correct_nums[-1])

            # 数値の近似比較
            return abs(gen_val - correct_val) < 1e-6

        except:
            # 文字列の完全一致にフォールバック
            return generated_answer.strip().lower() == correct_answer.strip().lower()

    def evaluate_code_execution(self, generated_code: str, sample: Dict) -> bool:
        """
        コードの実行による評価

        Args:
            generated_code: 生成されたコード
            sample: テストサンプル

        Returns:
            実行成功かどうか
        """
        try:
            if sample["dataset"] == "humaneval":
                # HumanEvalのテスト実行
                test_code = f"{generated_code}\n\n{sample['test']}"

                # 一時ファイルでテスト実行
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(test_code)
                    temp_file = f.name

                try:
                    # タイムアウト付きで実行
                    result = subprocess.run([
                        sys.executable, temp_file
                    ], capture_output=True, timeout=5.0, text=True)

                    success = result.returncode == 0
                    return success

                except subprocess.TimeoutExpired:
                    return False
                except Exception:
                    return False
                finally:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)

            elif sample["dataset"] == "mbpp":
                # MBPPのテスト実行
                test_cases = sample["test_list"]

                # 各テストケースを実行
                success_count = 0
                for test_case in test_cases[:3]:  # 最初の3つのテストケースのみ
                    try:
                        test_code = f"{generated_code}\n\n{test_case}"

                        # タイムアウト付きで実行
                        local_vars = {}

                        # シグナルでタイムアウト設定
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(3)  # 3秒タイムアウト

                        try:
                            exec(test_code, {}, local_vars)
                            success_count += 1
                        except TimeoutError:
                            pass
                        except Exception:
                            pass
                        finally:
                            signal.alarm(0)  # タイマーをリセット

                    except Exception:
                        continue

                # 2/3以上成功したらOK
                return success_count >= max(1, len(test_cases[:3]) * 0.67)

        except Exception as e:
            return False

        return False

    def run_single_evaluation(self, sample: Dict, method: str,
                              scheduler_config: Dict = None,
                              gen_length: int = 256) -> Dict[str, Any]:
        """
        単一サンプルの評価を実行

        Args:
            sample: 評価サンプル
            method: 評価手法 ("adaptive" or "static")
            scheduler_config: スケジューラー設定
            gen_length: 生成長

        Returns:
            評価結果
        """
        prompt_text = self.format_prompt(sample)
        prompt_ids = self.tokenizer.encode(
            prompt_text, return_tensors='pt').to(self.device)

        start_time = time.time()

        try:
            if method == "adaptive":
                # アダプティブスケジューリング
                output, metrics = generate_with_adaptive_scheduling(
                    model=self.model,
                    prompt=prompt_ids,
                    gen_length=gen_length,
                    enable_tiered_cache=True,
                    scheduler_config=scheduler_config,
                    verbose=False
                )

                nfe = metrics['nfe']
                adaptations = metrics['total_adaptations']
                mode_changes = metrics['total_adaptations']
                avg_block_size = metrics.get('avg_block_size', 0)

            else:  # static
                # 静的デュアルキャッシュ
                output, nfe = generate_with_dual_cache(
                    model=self.model,
                    prompt=prompt_ids,
                    steps=128,
                    gen_length=gen_length,
                    block_length=32,
                    threshold=0.8
                )

                adaptations = 0
                mode_changes = 0
                avg_block_size = 32

        except Exception as e:
            print(f"❌ 生成エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "generation_time": time.time() - start_time
            }

        generation_time = time.time() - start_time

        # 生成テキストをデコード
        generated_text = self.tokenizer.decode(
            output[0, prompt_ids.shape[1]:],
            skip_special_tokens=True
        )

        # 答えを抽出
        extracted_answer = self.extract_answer(generated_text, sample)

        # 正確性を評価
        if sample["type"] == "math":
            correct_answer = sample["answer"]
            is_correct = self.evaluate_math_answer(
                extracted_answer, correct_answer)
        elif sample["type"] == "code":
            is_correct = self.evaluate_code_execution(extracted_answer, sample)
        else:
            is_correct = False

        return {
            "success": True,
            "dataset": sample["dataset"],
            "sample_id": sample["id"],
            "method": method,
            "generation_time": generation_time,
            "nfe": nfe,
            "adaptations": adaptations,
            "mode_changes": mode_changes,
            "avg_block_size": avg_block_size,
            "is_correct": is_correct,
            "generated_text": generated_text[:500],  # 最初の500文字のみ保存
            "extracted_answer": extracted_answer[:200]  # 最初の200文字のみ保存
        }

    def run_comprehensive_benchmark(self,
                                    samples_per_dataset: int = 25,
                                    gen_length: int = 256,
                                    scheduler_config: Dict = None) -> Dict[str, Any]:
        """
        包括的ベンチマーク実行

        Args:
            samples_per_dataset: 各データセットのサンプル数
            gen_length: 生成長
            scheduler_config: スケジューラー設定

        Returns:
            ベンチマーク結果
        """
        print(f"\n🚀 包括的ベンチマーク開始")
        print(f"   各データセット: {samples_per_dataset}サンプル")
        print(f"   生成長: {gen_length}")
        print(f"   合計予定サンプル: {samples_per_dataset * 4}")

        # デフォルトスケジューラー設定
        if scheduler_config is None:
            scheduler_config = {
                'to_quality_threshold': 0.80,
                'to_efficiency_threshold': 0.95,
                'confidence_window_size': 2,
                'high_efficiency_params': {'block_size': 32, 'threshold': 0.75},
                'high_quality_params': {'block_size': 8, 'threshold': 0.95}
            }

        # データセット読み込み
        datasets = self.load_datasets(samples_per_dataset)

        # 全サンプルをまとめる
        all_samples = []
        for dataset_name, samples in datasets.items():
            all_samples.extend(samples)

        print(f"📊 実際のサンプル数: {len(all_samples)}")

        # 評価手法
        methods = ["adaptive", "static"]

        results = {
            "summary": {},
            "detailed_results": [],
            "dataset_breakdown": {},
            "config": {
                "samples_per_dataset": samples_per_dataset,
                "gen_length": gen_length,
                "scheduler_config": scheduler_config,
                "total_samples": len(all_samples)
            }
        }

        # 各手法で評価
        for method in methods:
            print(f"\n📊 {method.upper()}手法評価中...")
            method_results = []

            # プログレスバー付きで実行
            for sample in tqdm(all_samples, desc=f"{method.upper()}"):
                try:
                    result = self.run_single_evaluation(
                        sample=sample,
                        method=method,
                        scheduler_config=scheduler_config,
                        gen_length=gen_length
                    )
                    method_results.append(result)

                except Exception as e:
                    print(f"❌ サンプル{sample['id']}でエラー: {e}")
                    error_result = {
                        "success": False,
                        "dataset": sample["dataset"],
                        "sample_id": sample["id"],
                        "method": method,
                        "error": str(e)
                    }
                    method_results.append(error_result)

            # 結果を詳細結果に追加
            results["detailed_results"].extend(method_results)

            # 手法別サマリーを計算
            successful_results = [
                r for r in method_results if r.get("success", False)]

            if successful_results:
                total_time = sum(r["generation_time"]
                                 for r in successful_results)
                total_nfe = sum(r["nfe"] for r in successful_results)
                total_correct = sum(r["is_correct"]
                                    for r in successful_results)

                if method == "adaptive":
                    total_adaptations = sum(r["adaptations"]
                                            for r in successful_results)
                    avg_block_size = np.mean(
                        [r["avg_block_size"] for r in successful_results])
                else:
                    total_adaptations = 0
                    avg_block_size = 32

                results["summary"][method] = {
                    "total_samples": len(successful_results),
                    "total_time": total_time,
                    "avg_time_per_sample": total_time / len(successful_results),
                    "total_nfe": total_nfe,
                    "avg_nfe_per_sample": total_nfe / len(successful_results),
                    "accuracy": total_correct / len(successful_results),
                    "total_adaptations": total_adaptations,
                    "avg_block_size": avg_block_size
                }

                print(f"✅ {method.upper()}完了:")
                print(f"   成功サンプル: {len(successful_results)}")
                print(f"   平均時間: {total_time/len(successful_results):.2f}秒")
                print(
                    f"   精度: {total_correct/len(successful_results)*100:.1f}%")
                print(f"   平均NFE: {total_nfe/len(successful_results):.1f}")

        # データセット別分析
        for dataset_name in datasets.keys():
            dataset_results = [r for r in results["detailed_results"]
                               if r.get("dataset") == dataset_name and r.get("success", False)]

            if not dataset_results:
                continue

            results["dataset_breakdown"][dataset_name] = {}

            for method in methods:
                method_dataset_results = [
                    r for r in dataset_results if r["method"] == method]

                if method_dataset_results:
                    total_time = sum(r["generation_time"]
                                     for r in method_dataset_results)
                    total_correct = sum(r["is_correct"]
                                        for r in method_dataset_results)

                    results["dataset_breakdown"][dataset_name][method] = {
                        "samples": len(method_dataset_results),
                        "avg_time": total_time / len(method_dataset_results),
                        "accuracy": total_correct / len(method_dataset_results)
                    }

        return results

    def print_results_summary(self, results: Dict[str, Any]):
        """
        結果サマリーを表示

        Args:
            results: ベンチマーク結果
        """
        print(f"\n{'='*60}")
        print(f"📊 ベンチマーク結果サマリー")
        print(f"{'='*60}")

        summary = results["summary"]

        if "adaptive" in summary and "static" in summary:
            adaptive = summary["adaptive"]
            static = summary["static"]

            print(f"\n🔄 アダプティブ vs 静的比較:")
            print(f"{'メトリック':<20} {'アダプティブ':<15} {'静的':<15} {'改善率':<10}")
            print("-" * 65)

            # 時間比較
            time_improvement = (
                static["avg_time_per_sample"] - adaptive["avg_time_per_sample"]) / static["avg_time_per_sample"] * 100
            print(
                f"{'平均時間(秒)':<20} {adaptive['avg_time_per_sample']:<15.2f} {static['avg_time_per_sample']:<15.2f} {time_improvement:>+8.1f}%")

            # NFE比較
            nfe_improvement = (
                static["avg_nfe_per_sample"] - adaptive["avg_nfe_per_sample"]) / static["avg_nfe_per_sample"] * 100
            print(
                f"{'平均NFE':<20} {adaptive['avg_nfe_per_sample']:<15.1f} {static['avg_nfe_per_sample']:<15.1f} {nfe_improvement:>+8.1f}%")

            # 精度比較
            accuracy_improvement = (
                adaptive["accuracy"] - static["accuracy"]) * 100
            print(
                f"{'精度':<20} {adaptive['accuracy']*100:<15.1f}% {static['accuracy']*100:<15.1f}% {accuracy_improvement:>+8.1f}%")

            print(f"{'適応回数':<20} {adaptive['total_adaptations']:<15}")
            print(
                f"{'平均ブロックサイズ':<20} {adaptive['avg_block_size']:<15.1f} {static['avg_block_size']:<15.1f}")

        # データセット別結果
        if "dataset_breakdown" in results:
            print(f"\n📚 データセット別結果:")
            breakdown = results["dataset_breakdown"]

            for dataset_name, dataset_results in breakdown.items():
                if "adaptive" in dataset_results and "static" in dataset_results:
                    adaptive = dataset_results["adaptive"]
                    static = dataset_results["static"]

                    print(f"\n{dataset_name.upper()}:")
                    print(
                        f"  アダプティブ: 精度={adaptive['accuracy']*100:.1f}%, 時間={adaptive['avg_time']:.2f}s")
                    print(
                        f"  静的:        精度={static['accuracy']*100:.1f}%, 時間={static['avg_time']:.2f}s")

    def save_results(self, results: Dict[str, Any], output_dir: str = "benchmark_results"):
        """
        結果をファイルに保存

        Args:
            results: ベンチマーク結果
            output_dir: 出力ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 詳細結果保存
        results_file = output_path / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # CSVサマリー保存
        summary_file = output_path / f"benchmark_summary_{timestamp}.csv"
        with open(summary_file, 'w') as f:
            f.write("method,dataset,samples,avg_time,accuracy,avg_nfe,adaptations\n")

            for dataset_name, dataset_results in results.get("dataset_breakdown", {}).items():
                for method, method_results in dataset_results.items():
                    nfe = results["summary"][method]["avg_nfe_per_sample"] if method in results["summary"] else 0
                    adaptations = results["summary"][method]["total_adaptations"] if method in results["summary"] else 0

                    f.write(f"{method},{dataset_name},{method_results['samples']},"
                            f"{method_results['avg_time']:.3f},{method_results['accuracy']:.3f},"
                            f"{nfe:.1f},{adaptations}\n")

        print(f"💾 結果保存完了:")
        print(f"   詳細結果: {results_file}")
        print(f"   CSVサマリー: {summary_file}")


def main():
    """
    使用例:

    # 基本ベンチマーク (各25サンプル)
    python benchmark_runner.py --samples 25

    # 高速テスト (各5サンプル)
    python benchmark_runner.py --samples 5 --gen-length 128

    # カスタムスケジューラー設定
    python benchmark_runner.py --samples 25 --to-quality-threshold 0.75 --efficiency-block-size 16
    """
    parser = argparse.ArgumentParser(
        description="Adaptive Scheduling Benchmark Runner")

    # 基本パラメータ
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct",
                        help="モデル名")
    parser.add_argument("--device", default="auto",
                        help="使用デバイス")
    parser.add_argument("--samples", type=int, default=25,
                        help="各データセットのサンプル数")
    parser.add_argument("--gen-length", type=int, default=256,
                        help="生成長")

    # スケジューラー設定
    scheduler_group = parser.add_argument_group('scheduler', 'スケジューラー設定')
    scheduler_group.add_argument("--to-quality-threshold", type=float, default=0.80,
                                 help="品質モードへの切り替え閾値")
    scheduler_group.add_argument("--to-efficiency-threshold", type=float, default=0.95,
                                 help="効率モードへの切り替え閾値")
    scheduler_group.add_argument("--confidence-window-size", type=int, default=2,
                                 help="信頼度ウィンドウサイズ")
    scheduler_group.add_argument("--efficiency-block-size", type=int, default=32,
                                 help="効率モードのブロックサイズ")
    scheduler_group.add_argument("--quality-block-size", type=int, default=8,
                                 help="品質モードのブロックサイズ")
    scheduler_group.add_argument("--efficiency-threshold", type=float, default=0.75,
                                 help="効率モードの信頼度閾値")
    scheduler_group.add_argument("--quality-threshold", type=float, default=0.95,
                                 help="品質モードの信頼度閾値")

    # 出力設定
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="結果出力ディレクトリ")

    args = parser.parse_args()

    # スケジューラー設定構築
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

    print(f"🔧 設定:")
    print(f"   各データセット: {args.samples}サンプル")
    print(f"   生成長: {args.gen_length}")
    print(f"   スケジューラー設定: {scheduler_config}")

    # ベンチマーク実行
    runner = BenchmarkRunner(model_name=args.model, device=args.device)

    try:
        results = runner.run_comprehensive_benchmark(
            samples_per_dataset=args.samples,
            gen_length=args.gen_length,
            scheduler_config=scheduler_config
        )

        # 結果表示
        runner.print_results_summary(results)

        # 結果保存
        runner.save_results(results, args.output_dir)

        print(f"\n🎉 ベンチマーク完了!")

    except Exception as e:
        print(f"❌ ベンチマーク実行中にエラーが発生: {e}")
        raise


if __name__ == "__main__":
    main()
