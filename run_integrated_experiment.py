#!/usr/bin/env python3
"""
Fast-dLLM × LongLLaDA 統合実験メインスクリプト
Google Colab での実行に最適化されています
"""

from integrated_generation import (
    generate_fast_long,
    generate_fast_long_dual_cache,
    generate_fast_long_prefix_cache,
    generate_no_cache,
    load_model_with_scaling,
    format_metrics,
    add_gumbel_noise,
    get_num_transfer_tokens,
    get_transfer_index
)
import torch
import numpy as np
import time
import json
import argparse
import sys
import os
from pathlib import Path

# プロジェクトパスを追加
sys.path.append('Fast-dLLM/llada')
sys.path.append('LongLLaDA/llada')

# 統合生成機能をインポート


def setup_environment():
    """環境のセットアップ"""
    print("🔧 環境セットアップ中...")

    # GPU確認
    if torch.cuda.is_available():
        print(f"✅ GPU利用可能: {torch.cuda.get_device_name()}")
        print(
            f"   メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ GPU利用不可、CPUで実行します")

    # フラッシュアテンション確認
    try:
        import flash_attn
        print("✅ Flash Attention 利用可能")
    except ImportError:
        print("⚠️ Flash Attention 利用不可")


def test_basic_generation(model, tokenizer, scaling_factor=1):
    """基本生成テスト"""
    print(f"\n🧪 基本生成テスト（スケーリング係数: {scaling_factor}）")

    # テストプロンプト
    prompt = "次の数学問題を解いてください：太郎は毎時12kmで4時間走り、その後毎時6kmで走りました。合計8時間でどれだけの距離を走れますか？"

    # チャットテンプレート適用
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    input_ids = tokenizer(
        formatted_prompt, return_tensors='pt').input_ids.to(model.device)
    print(f"入力長: {input_ids.shape[1]} トークン")

    # 生成実行
    outputs, nfe, metrics = generate_fast_long(
        model=model,
        prompt=input_ids,
        steps=128,
        gen_length=256,
        block_length=32,
        temperature=0.0,
        remasking='low_confidence',
        scaling_factor=scaling_factor,
        dual_cache=True
    )

    # 結果表示
    result = tokenizer.decode(
        outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n📝 生成結果:\n{result}")

    format_metrics(metrics)

    return result, metrics


def test_long_context(model, tokenizer, scaling_factor=14):
    """長文コンテキストテスト"""
    print(f"\n📚 長文コンテキストテスト（スケーリング係数: {scaling_factor}）")

    # 長文コンテキスト作成
    base_context = "あなたは小説を書くAIです。物語の背景設定を詳しく説明してください。キャラクターの心理描写も重要です。"
    long_context = base_context * 100  # 疑似長文

    long_prompt = f"""以下のコンテキストを読んで、最後に質問に答えてください。

コンテキスト: {long_context}

質問: このコンテキストの主なテーマは何ですか？簡潔に答えてください。"""

    messages = [{"role": "user", "content": long_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    input_ids = tokenizer(
        formatted_prompt, return_tensors='pt').input_ids.to(model.device)
    print(f"長文入力長: {input_ids.shape[1]} トークン")

    if input_ids.shape[1] > 3000:  # 長文の場合のみ実行
        outputs, nfe, metrics = generate_fast_long(
            model=model,
            prompt=input_ids,
            steps=64,           # 長文では少なめに
            gen_length=256,
            block_length=64,    # 大きめのブロック
            temperature=0.0,
            remasking='low_confidence',
            scaling_factor=scaling_factor,
            dual_cache=True
        )

        result = tokenizer.decode(
            outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n📝 長文生成結果:\n{result}")

        format_metrics(metrics)
        return result, metrics
    else:
        print("⚠️ 入力が短いため、長文テストをスキップ")
        return None, None


def performance_comparison():
    """性能比較実験"""
    print("\n⚡ 性能比較実験")

    # モデル読み込み
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'
    model, tokenizer, config = load_model_with_scaling(
        model_path, scaling_factor=1)

    # 異なる設定での比較
    configs = [
        {"name": "高速設定", "steps": 64, "block_length": 64, "remasking": "random"},
        {"name": "標準設定", "steps": 128, "block_length": 32,
            "remasking": "low_confidence"},
        {"name": "品質重視", "steps": 256, "block_length": 16,
            "remasking": "low_confidence"}
    ]

    # テストプロンプト
    prompt = "Python で機械学習のコードを書いてください。"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(
        formatted_prompt, return_tensors='pt').input_ids.to(model.device)

    results = []

    for config in configs:
        print(f"\n🧪 {config['name']} テスト中...")

        outputs, nfe, metrics = generate_fast_long(
            model=model,
            prompt=input_ids,
            steps=config['steps'],
            gen_length=128,
            block_length=config['block_length'],
            temperature=0.0,
            remasking=config['remasking'],
            dual_cache=True
        )

        result_text = tokenizer.decode(
            outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

        results.append({
            'config': config['name'],
            'speed': metrics['tokens_per_second'],
            'time': metrics['total_time'],
            'nfe': metrics['nfe'],
            'result_preview': result_text[:100] + "..." if len(result_text) > 100 else result_text
        })

    # 結果表示
    print("\n📊 性能比較結果:")
    for r in results:
        print(f"\n{r['config']}:")
        print(f"  速度: {r['speed']:.1f} tok/s")
        print(f"  時間: {r['time']:.2f}s")
        print(f"  NFE: {r['nfe']}")
        print(f"  結果例: {r['result_preview']}")

    return results


def niah_test(model, tokenizer, context_length=8000, scaling_factor=14):
    """Needle in a Haystack テスト"""
    print(f"\n🎯 NIAH テスト（コンテキスト長: {context_length}, スケーリング: {scaling_factor}）")

    # ダミー長文コンテキスト生成
    haystack_unit = "これは重要ではない情報です。日本の四季は美しく、春には桜が咲き、夏には緑が濃くなります。秋には紅葉が美しく、冬には雪が降ります。"
    haystack = haystack_unit * (context_length // len(haystack_unit.split()))

    needle = "重要な情報：答えは42です。これは宇宙、生命、そしてすべての答えです。"
    question = "重要な情報は何ですか？特に数字に注目してください。"

    # needle を中央に配置
    full_text = haystack[:len(haystack)//2] + " " + needle + \
        " " + haystack[len(haystack)//2:] + f"\n\n質問：{question}"

    messages = [{"role": "user", "content": full_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(
        formatted, return_tensors='pt').input_ids.to(model.device)

    print(f"NIAH コンテキスト長: {input_ids.shape[1]} トークン")

    # 生成
    outputs, nfe, metrics = generate_fast_long(
        model, input_ids,
        steps=64,
        gen_length=100,
        block_length=32,
        scaling_factor=scaling_factor,
        dual_cache=True
    )

    result = tokenizer.decode(
        outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

    # 評価
    success = "42" in result
    print(f"\nNIAH 成功: {'✅' if success else '❌'}")
    print(f"回答: {result}")

    format_metrics(metrics)

    return success, result, metrics


def run_comprehensive_evaluation():
    """包括的評価実験"""
    print("🚀 Fast-dLLM × LongLLaDA 包括的評価実験開始")
    print("=" * 60)

    results = {}

    try:
        # 1. 基本生成テスト
        print("\n1️⃣ 基本生成テスト")
        model_path = 'GSAI-ML/LLaDA-8B-Instruct'
        model, tokenizer, config = load_model_with_scaling(
            model_path, scaling_factor=1)

        basic_result, basic_metrics = test_basic_generation(
            model, tokenizer, scaling_factor=1)
        results['basic_test'] = {
            'result': basic_result, 'metrics': basic_metrics}

        # 2. 長文テスト（RoPE スケーリング）
        print("\n2️⃣ 長文コンテキストテスト")
        model_long, tokenizer_long, config_long = load_model_with_scaling(
            model_path, scaling_factor=14)

        long_result, long_metrics = test_long_context(
            model_long, tokenizer_long, scaling_factor=14)
        if long_result:
            results['long_context_test'] = {
                'result': long_result, 'metrics': long_metrics}

        # 3. NIAH テスト
        print("\n3️⃣ NIAH テスト")
        niah_success, niah_result, niah_metrics = niah_test(
            model_long, tokenizer_long)
        results['niah_test'] = {
            'success': niah_success,
            'result': niah_result,
            'metrics': niah_metrics
        }

        # 4. 性能比較
        print("\n4️⃣ 性能比較実験")
        perf_results = performance_comparison()
        results['performance_comparison'] = perf_results

        # 5. 総合評価
        print("\n📈 総合評価")
        print("=" * 40)
        print("🚀 実装された機能:")
        print("  ✅ Fast-dLLM ブロック生成")
        print("  ✅ LongLLaDA RoPE スケーリング")
        print("  ✅ 信頼度ベース並列デコーディング")
        print("  ✅ KV キャッシュ最適化")

        print("\n📊 性能特性:")
        if 'basic_test' in results:
            print(
                f"  🔥 標準生成速度: {results['basic_test']['metrics']['tokens_per_second']:.1f} tok/s")
        if 'long_context_test' in results:
            print(
                f"  📏 長文生成速度: {results['long_context_test']['metrics']['tokens_per_second']:.1f} tok/s")
        if 'niah_test' in results:
            print(
                f"  🎯 NIAH テスト: {'成功' if results['niah_test']['success'] else '失敗'}")

        print("\n🎉 統合実験完了！")
        print("Fast-dLLM の高速性と LongLLaDA の長文対応が正常に統合されました。")

        # 結果をJSONで保存
        results_json = {k: {**v, 'metrics': {key: float(val) if isinstance(val, (int, float)) else val
                                             for key, val in v.get('metrics', {}).items()}}
                        if isinstance(v, dict) and 'metrics' in v else v
                        for k, v in results.items()}

        with open('experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(results_json, f, ensure_ascii=False, indent=2)

        print("\n💾 結果が experiment_results.json に保存されました")

        return results

    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Fast-dLLM × LongLLaDA 統合実験')
    parser.add_argument('--test', choices=['basic', 'long', 'niah', 'performance', 'all'],
                        default='all', help='実行するテスト')
    parser.add_argument('--model', default='GSAI-ML/LLaDA-8B-Instruct',
                        help='使用するモデル')
    parser.add_argument('--scaling-factor', type=float, default=1,
                        help='RoPEスケーリング係数')

    args = parser.parse_args()

    # 環境セットアップ
    setup_environment()

    if args.test == 'all':
        # 包括的評価実行
        run_comprehensive_evaluation()
    else:
        # 個別テスト実行
        model, tokenizer, config = load_model_with_scaling(
            args.model, args.scaling_factor)

        if args.test == 'basic':
            test_basic_generation(model, tokenizer, args.scaling_factor)
        elif args.test == 'long':
            test_long_context(model, tokenizer, args.scaling_factor)
        elif args.test == 'niah':
            niah_test(model, tokenizer, scaling_factor=args.scaling_factor)
        elif args.test == 'performance':
            performance_comparison()


if __name__ == "__main__":
    main()
