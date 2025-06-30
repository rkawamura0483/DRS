#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GTS制御サンプリング vs Dual Cache 比較評価スクリプト

Phase 1B: Preliminary GTS Validation
- 複数プロンプトでの品質・速度比較
- 実行時間とNFE効率の測定
- GTSスコアと生成品質の相関分析
"""

import torch
import time
import json
from typing import List, Dict, Any
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate_with_dual_cache
from sampler.gts_controlled_sampler import generate_with_gts_controlled_sampling


def get_test_prompts() -> List[Dict[str, str]]:
    """
    評価用プロンプトセット（5個）
    異なるタスクタイプで多様性を確保
    """
    return [
        {
            "id": "math_word_problem",
            "prompt": "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
            "category": "数学的推論"
        },
        {
            "id": "logical_reasoning",
            "prompt": "If all cats are animals, and some animals are pets, can we conclude that some cats are pets?",
            "category": "論理的推論"
        },
        {
            "id": "code_generation",
            "prompt": "Write a Python function that calculates the factorial of a number using recursion.",
            "category": "コード生成"
        },
        {
            "id": "text_completion",
            "prompt": "The benefits of renewable energy include reducing greenhouse gas emissions, decreasing dependence on fossil fuels, and",
            "category": "文章補完"
        },
        {
            "id": "problem_solving",
            "prompt": "A baker has 120 cupcakes to pack into boxes. Each box can hold 8 cupcakes. How many full boxes can the baker make, and how many cupcakes will be left over?",
            "category": "問題解決"
        }
    ]


def evaluate_single_prompt(
    model,
    tokenizer,
    prompt_data: Dict[str, str],
    gen_length: int = 128,
    gts_threshold: float = 0.8,
    max_iterations: int = 3,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    単一プロンプトでの比較評価を実行
    """
    prompt = prompt_data["prompt"]
    prompt_id = prompt_data["id"]
    category = prompt_data["category"]

    print(f"\n{'='*60}")
    print(f"📝 評価中: {prompt_id} ({category})")
    print(f"プロンプト: {prompt[:60]}...")
    print(f"{'='*60}")

    # プロンプト処理
    m = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    results = {
        'prompt_id': prompt_id,
        'category': category,
        'original_prompt': prompt,
        'formatted_prompt': formatted_prompt,
        'methods': {}
    }

    # 1. Dual Cache 生成
    print("\n🚀 Dual Cache サンプリング...")
    start_time = time.time()
    try:
        dual_output, dual_nfe = generate_with_dual_cache(
            model, input_ids,
            steps=128, gen_length=gen_length,
            block_length=32, temperature=0.,
            remasking='low_confidence'
        )
        dual_time = time.time() - start_time
        dual_text = tokenizer.batch_decode(
            dual_output[:, input_ids.shape[1]:],
            skip_special_tokens=True)[0]

        results['methods']['dual_cache'] = {
            'success': True,
            'nfe': dual_nfe,
            'time': dual_time,
            'tokens_per_second': gen_length / dual_time,
            'nfe_efficiency': gen_length / dual_nfe,
            'generated_text': dual_text,
            'output_length': len(dual_text)
        }

        print(f"✅ 完了: NFE={dual_nfe}, 時間={dual_time:.2f}s")

    except Exception as e:
        print(f"❌ エラー: {e}")
        results['methods']['dual_cache'] = {'success': False, 'error': str(e)}

    # 2. GTS制御サンプリング
    print("\n🎯 GTS制御サンプリング...")
    start_time = time.time()
    try:
        gts_output, gts_metrics = generate_with_gts_controlled_sampling(
            model, input_ids,
            gen_length=gen_length,
            parallel_k=32,  # より小さなブロックで詳細な制御
            gts_threshold=gts_threshold,
            max_iterations=max_iterations,
            steps_per_iteration=16,  # より多くのdiffusion stepsを実行
            gts_metric_type='basic',  # 明示的にbasicを指定
            temperature=0.1,  # 小さな温度でより安定した生成
            verbose=True  # デバッグ用に詳細ログを有効化
        )
        gts_time = time.time() - start_time
        gts_text = tokenizer.batch_decode(
            gts_output[:, input_ids.shape[1]:],
            skip_special_tokens=True)[0]

        # GTSメトリクス処理
        total_iterations = gts_metrics.get(
            'total_iterations', gts_metrics.get('iterations', 0))
        gts_scores = gts_metrics.get(
            'gts_scores_history', gts_metrics.get('gts_scores', []))
        final_gts = gts_scores[-1] if gts_scores else 1.0
        avg_gts = sum(gts_scores) / len(gts_scores) if gts_scores else 1.0

        results['methods']['gts'] = {
            'success': True,
            'nfe': gts_metrics['nfe'],
            'time': gts_time,
            'iterations': total_iterations,
            'tokens_per_second': gen_length / gts_time,
            'nfe_efficiency': gen_length / gts_metrics['nfe'],
            'final_gts_score': final_gts,
            'avg_gts_score': avg_gts,
            'gts_scores_history': gts_scores,
            'generated_text': gts_text,
            'output_length': len(gts_text),
            'full_metrics': gts_metrics
        }

        print(
            f"✅ 完了: NFE={gts_metrics['nfe']}, 時間={gts_time:.2f}s, 反復={total_iterations}, GTS={final_gts:.4f}")

    except Exception as e:
        print(f"❌ エラー: {e}")
        results['methods']['gts'] = {'success': False, 'error': str(e)}

    # 3. 比較分析
    if results['methods']['dual_cache']['success'] and results['methods']['gts']['success']:
        dual = results['methods']['dual_cache']
        gts = results['methods']['gts']

        results['comparison'] = {
            'speed_ratio': dual['time'] / gts['time'],  # >1 ならGTSが高速
            'nfe_ratio': dual['nfe'] / gts['nfe'],      # >1 ならGTSが効率的
            'time_difference': gts['time'] - dual['time'],
            'nfe_difference': gts['nfe'] - dual['nfe'],
            'text_length_difference': len(gts['generated_text']) - len(dual['generated_text'])
        }

        print(f"\n📊 比較結果:")
        print(f"   速度比 (Dual/GTS): {results['comparison']['speed_ratio']:.2f}")
        print(f"   NFE比 (Dual/GTS): {results['comparison']['nfe_ratio']:.2f}")
        print(f"   時間差: {results['comparison']['time_difference']:+.2f}s")
        print(f"   NFE差: {results['comparison']['nfe_difference']:+d}")

    return results


def run_evaluation(
    gen_length: int = 128,
    gts_threshold: float = 0.8,
    max_iterations: int = 3,
    device: str = 'cuda',
    output_file: str = 'gts_evaluation_results.json'
) -> Dict[str, Any]:
    """
    GTS評価実験のメイン実行関数
    """
    print("🔬 GTS制御サンプリング評価実験開始")
    print(f"   生成長: {gen_length}")
    print(f"   GTS閾値: {gts_threshold}")
    print(f"   最大反復: {max_iterations}")
    print(f"   デバイス: {device}")

    # モデルロード
    print(f"\n🤖 モデルロード中...")
    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True
    )

    # テストプロンプト取得
    test_prompts = get_test_prompts()

    # 全体結果初期化
    evaluation_results = {
        'experiment_config': {
            'gen_length': gen_length,
            'gts_threshold': gts_threshold,
            'max_iterations': max_iterations,
            'device': device,
            'model_name': 'GSAI-ML/LLaDA-8B-Instruct',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'prompts': [],
        'summary_statistics': {}
    }

    # 各プロンプトで評価実行
    for i, prompt_data in enumerate(test_prompts):
        print(f"\n🎯 進行状況: {i+1}/{len(test_prompts)}")
        result = evaluate_single_prompt(
            model, tokenizer, prompt_data,
            gen_length=gen_length,
            gts_threshold=gts_threshold,
            max_iterations=max_iterations,
            device=device
        )
        evaluation_results['prompts'].append(result)

    # 統計サマリー計算
    evaluation_results['summary_statistics'] = calculate_summary_statistics(
        evaluation_results['prompts']
    )

    # 結果保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 結果を {output_file} に保存しました")

    # サマリー表示
    display_summary(evaluation_results)

    return evaluation_results


def calculate_summary_statistics(prompt_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    全プロンプトの統計サマリーを計算
    """
    successful_results = [
        r for r in prompt_results
        if r['methods'].get('dual_cache', {}).get('success') and
        r['methods'].get('gts', {}).get('success')
    ]

    if not successful_results:
        return {'error': 'no_successful_comparisons'}

    # 時間統計
    dual_times = [r['methods']['dual_cache']['time']
                  for r in successful_results]
    gts_times = [r['methods']['gts']['time'] for r in successful_results]

    # NFE統計
    dual_nfes = [r['methods']['dual_cache']['nfe'] for r in successful_results]
    gts_nfes = [r['methods']['gts']['nfe'] for r in successful_results]

    # GTS統計
    gts_scores = [r['methods']['gts']['final_gts_score']
                  for r in successful_results]
    gts_iterations = [r['methods']['gts']['iterations']
                      for r in successful_results]

    return {
        'num_successful_comparisons': len(successful_results),
        'timing': {
            'dual_cache_avg': sum(dual_times) / len(dual_times),
            'gts_avg': sum(gts_times) / len(gts_times),
            'gts_speedup_ratio': sum(dual_times) / sum(gts_times),
            'time_savings_avg': (sum(dual_times) - sum(gts_times)) / len(dual_times)
        },
        'efficiency': {
            'dual_cache_nfe_avg': sum(dual_nfes) / len(dual_nfes),
            'gts_nfe_avg': sum(gts_nfes) / len(gts_nfes),
            'nfe_efficiency_ratio': sum(dual_nfes) / sum(gts_nfes),
            'nfe_savings_avg': (sum(dual_nfes) - sum(gts_nfes)) / len(dual_nfes)
        },
        'gts_quality': {
            'avg_final_gts_score': sum(gts_scores) / len(gts_scores),
            'min_gts_score': min(gts_scores),
            'max_gts_score': max(gts_scores),
            'avg_iterations': sum(gts_iterations) / len(gts_iterations),
            'max_iterations': max(gts_iterations)
        }
    }


def display_summary(evaluation_results: Dict[str, Any]) -> None:
    """
    評価結果の見やすいサマリーを表示
    """
    stats = evaluation_results['summary_statistics']

    if 'error' in stats:
        print("❌ 統計計算でエラーが発生しました")
        return

    print(f"\n{'='*60}")
    print(f"📈 GTS制御サンプリング評価結果サマリー")
    print(f"{'='*60}")

    print(f"\n🕒 実行時間比較:")
    print(f"   Dual Cache平均: {stats['timing']['dual_cache_avg']:.2f}秒")
    print(f"   GTS制御平均:   {stats['timing']['gts_avg']:.2f}秒")
    print(f"   GTSスピードアップ率: {stats['timing']['gts_speedup_ratio']:.2f}x")
    if stats['timing']['gts_speedup_ratio'] > 1:
        print(f"   ✅ GTSが {stats['timing']['time_savings_avg']:.2f}秒高速")
    else:
        print(f"   ⚠️ GTSが {-stats['timing']['time_savings_avg']:.2f}秒低速")

    print(f"\n⚡ NFE効率比較:")
    print(
        f"   Dual Cache平均: {stats['efficiency']['dual_cache_nfe_avg']:.1f} NFE")
    print(f"   GTS制御平均:   {stats['efficiency']['gts_nfe_avg']:.1f} NFE")
    print(f"   GTS効率向上率: {stats['efficiency']['nfe_efficiency_ratio']:.2f}x")
    if stats['efficiency']['nfe_efficiency_ratio'] > 1:
        print(f"   ✅ GTSが {stats['efficiency']['nfe_savings_avg']:.1f} NFE効率的")
    else:
        print(
            f"   ⚠️ GTSが {-stats['efficiency']['nfe_savings_avg']:.1f} NFE非効率")

    print(f"\n🎯 GTS品質統計:")
    print(f"   平均GTSスコア: {stats['gts_quality']['avg_final_gts_score']:.4f}")
    print(
        f"   GTS範囲:      {stats['gts_quality']['min_gts_score']:.4f} - {stats['gts_quality']['max_gts_score']:.4f}")
    print(f"   平均反復回数:  {stats['gts_quality']['avg_iterations']:.1f}")
    print(f"   最大反復回数:  {stats['gts_quality']['max_iterations']}")

    print(f"\n📊 評価対象: {stats['num_successful_comparisons']}プロンプト")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GTS制御サンプリング評価実験')
    parser.add_argument('--gen_length', type=int, default=128, help='生成長')
    parser.add_argument('--gts_threshold', type=float,
                        default=0.8, help='GTS閾値')
    parser.add_argument('--max_iterations', type=int, default=3, help='最大反復回数')
    parser.add_argument('--device', type=str, default='cuda', help='デバイス')
    parser.add_argument('--output', type=str,
                        default='gts_evaluation_results.json', help='出力ファイル名')

    args = parser.parse_args()

    results = run_evaluation(
        gen_length=args.gen_length,
        gts_threshold=args.gts_threshold,
        max_iterations=args.max_iterations,
        device=args.device,
        output_file=args.output
    )
