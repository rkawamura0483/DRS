#!/usr/bin/env python3
"""
Fast-dLLM × LongLLaDA クイックスタートスクリプト
Google Colab での簡単な実行のためのデモ（修正版）
"""

from integrated_generation import (
    generate_fast_long,
    load_model_with_scaling,
    format_metrics,
    generate_fast_long_dual_cache,
    generate_fast_long_prefix_cache,
    generate_no_cache
)
import torch
import time
import sys
import os

# プロジェクトパスを追加
sys.path.append('Fast-dLLM/llada')
sys.path.append('LongLLaDA/llada')


def check_environment():
    """環境チェック"""
    print("🔍 環境チェック")
    print("=" * 30)

    # GPU確認
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ GPU利用不可")

    # PyTorch確認
    print(f"✅ PyTorch: {torch.__version__}")

    # Transformers確認
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers が見つかりません")

    # Flash Attention確認
    try:
        import flash_attn
        print("✅ Flash Attention 利用可能")
    except ImportError:
        print("⚠️ Flash Attention 利用不可")

    print()


def demo_basic_generation():
    """基本生成デモ（修正版）"""
    print("🚀 基本生成デモ（Fast-dLLM デュアルキャッシュ使用）")
    print("=" * 30)

    # モデル読み込み
    print("📥 モデル読み込み中...")
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'
    model, tokenizer, config = load_model_with_scaling(
        model_path, scaling_factor=1)

    # 簡単なテスト
    prompt = "こんにちは！あなたは何ができますか？"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    input_ids = tokenizer(
        formatted_prompt, return_tensors='pt').input_ids.to(model.device)

    print(f"📝 プロンプト: {prompt}")
    print(f"🔤 入力長: {input_ids.shape[1]} トークン")

    # 生成実行（デュアルキャッシュ使用）
    print("⚡ 生成中（Fast-dLLM デュアルキャッシュ）...")
    outputs, nfe, metrics = generate_fast_long_dual_cache(
        model=model,
        prompt=input_ids,
        steps=64,      # 高速化のため少なめ
        gen_length=128,
        block_length=32,
        temperature=0.0,
        remasking='low_confidence',
        scaling_factor=1
    )

    # 結果表示
    result = tokenizer.decode(
        outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
    print("\n📖 生成結果:")
    print("-" * 40)
    print(result)
    print("-" * 40)

    format_metrics(metrics)

    return model, tokenizer


def demo_long_context(model, tokenizer):
    """長文コンテキストデモ（修正版）"""
    print("\n📚 長文コンテキストデモ（RoPE スケーリング）")
    print("=" * 30)

    # 長文読み込み（RoPEスケーリング適用）
    print("🔧 RoPE スケーリング適用...")
    model_long, tokenizer_long, config_long = load_model_with_scaling(
        'GSAI-ML/LLaDA-8B-Instruct',
        scaling_factor=14  # 16k対応
    )

    # 長文プロンプト作成
    story_context = """昔々、ある美しい森に不思議な力を持つ魔法使いが住んでいました。
その魔法使いは動物たちと話すことができ、季節を操る力を持っていました。
春には花を咲かせ、夏には涼しい風を送り、秋には美しい紅葉を作り出し、冬には温かい雪を降らせました。
森の動物たちは皆、この優しい魔法使いを慕っていました。
ある日、遠い国から一人の旅人がこの森にやってきました。
旅人は道に迷い、疲れ果てていました。
魔法使いは旅人を助け、森の奥にある小さな小屋で休ませてあげました。
旅人は魔法使いの優しさに感動し、何かお礼をしたいと申し出ました。""" * 20  # 繰り返して長文化

    question = "この物語の主人公の特徴を3つ教えてください。"
    long_prompt = f"""以下の物語を読んで、質問に答えてください。

物語:
{story_context}

質問: {question}"""

    messages = [{"role": "user", "content": long_prompt}]
    formatted_prompt = tokenizer_long.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    input_ids = tokenizer_long(
        formatted_prompt, return_tensors='pt').input_ids.to(model_long.device)

    print(f"📏 長文入力長: {input_ids.shape[1]} トークン")

    if input_ids.shape[1] > 2000:  # 長文の場合
        print("⚡ 長文生成中（RoPEスケーリング + デュアルキャッシュ）...")
        outputs, nfe, metrics = generate_fast_long_dual_cache(
            model=model_long,
            prompt=input_ids,
            steps=32,       # 長文では更に少なめ
            gen_length=256,
            block_length=64,
            temperature=0.0,
            remasking='low_confidence',
            scaling_factor=14
        )

        result = tokenizer_long.decode(
            outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
        print("\n📖 長文生成結果:")
        print("-" * 40)
        print(result)
        print("-" * 40)

        format_metrics(metrics)
    else:
        print("⚠️ コンテキストが短いため長文テストをスキップ")


def demo_cache_comparison():
    """キャッシュ方式比較デモ"""
    print("\n💾 キャッシュ方式比較デモ")
    print("=" * 30)

    # モデル読み込み
    model, tokenizer, _ = load_model_with_scaling('GSAI-ML/LLaDA-8B-Instruct')

    prompt = "プログラミング初心者へのアドバイスを3つ教えてください。"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(
        formatted_prompt, return_tensors='pt').input_ids.to(model.device)

    cache_configs = [
        {
            "name": "🚫 キャッシュなし",
            "func": generate_no_cache,
            "steps": 64,
            "block_length": 32
        },
        {
            "name": "📋 プレフィックスキャッシュ",
            "func": generate_fast_long_prefix_cache,
            "steps": 64,
            "block_length": 32
        },
        {
            "name": "💎 デュアルキャッシュ",
            "func": generate_fast_long_dual_cache,
            "steps": 64,
            "block_length": 32
        }
    ]

    for config in cache_configs:
        print(f"\n{config['name']} テスト...")

        outputs, nfe, metrics = config['func'](
            model=model,
            prompt=input_ids,
            steps=config['steps'],
            gen_length=64,  # 短めで高速化
            block_length=config['block_length'],
            temperature=0.0,
            remasking='low_confidence'
        )

        result = tokenizer.decode(
            outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

        print(f"  ⏱️  時間: {metrics['total_time']:.2f}秒")
        print(f"  🚀 速度: {metrics['tokens_per_second']:.1f} tok/s")
        print(f"  🔄 NFE: {metrics['nfe']}")
        print(f"  💾 キャッシュヒット: {metrics['cache_hits']}")
        print(f"  📝 結果: {result[:80]}...")


def demo_speed_comparison():
    """速度比較デモ（修正版）"""
    print("\n⚡ 速度設定比較デモ")
    print("=" * 30)

    # モデル読み込み
    model, tokenizer, _ = load_model_with_scaling('GSAI-ML/LLaDA-8B-Instruct')

    prompt = "機械学習について簡潔に説明してください。"
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(
        formatted_prompt, return_tensors='pt').input_ids.to(model.device)

    configs = [
        {"name": "🏃 超高速", "steps": 32, "block_length": 64, "remasking": "random"},
        {"name": "⚡ 高速", "steps": 64, "block_length": 32, "remasking": "random"},
        {"name": "🎯 標準", "steps": 64, "block_length": 32,
            "remasking": "low_confidence"},
        {"name": "🎨 品質重視", "steps": 128, "block_length": 16,
            "remasking": "low_confidence"}
    ]

    for config in configs:
        print(f"\n{config['name']} 設定テスト...")

        outputs, nfe, metrics = generate_fast_long_dual_cache(
            model=model,
            prompt=input_ids,
            steps=config['steps'],
            gen_length=64,  # 短めで高速化
            block_length=config['block_length'],
            temperature=0.0,
            remasking=config['remasking']
        )

        result = tokenizer.decode(
            outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

        print(f"  ⏱️  時間: {metrics['total_time']:.2f}秒")
        print(f"  🚀 速度: {metrics['tokens_per_second']:.1f} tok/s")
        print(f"  🔄 NFE: {metrics['nfe']}")
        print(f"  📝 結果: {result[:80]}...")


def demo_niah_simple():
    """簡単なNIAHテスト（修正版）"""
    print("\n🎯 簡単なNIAHテスト（RoPE スケーリング）")
    print("=" * 30)

    # 長文対応モデル
    model, tokenizer, _ = load_model_with_scaling(
        'GSAI-ML/LLaDA-8B-Instruct', scaling_factor=14)

    # 簡単なNIAH作成
    haystack = "今日は良い天気です。空が青くて雲が白いです。鳥たちが空を飛んでいます。" * 100
    needle = "重要：パスワードは1234です。"
    question = "パスワードは何ですか？"

    full_text = haystack + needle + haystack + f"\n\n質問: {question}"

    messages = [{"role": "user", "content": full_text}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(
        formatted, return_tensors='pt').input_ids.to(model.device)

    print(f"📏 コンテキスト長: {input_ids.shape[1]} トークン")

    outputs, nfe, metrics = generate_fast_long_dual_cache(
        model, input_ids,
        steps=32,
        gen_length=50,
        block_length=32,
        scaling_factor=14
    )

    result = tokenizer.decode(
        outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

    success = "1234" in result
    print(f"\n🎯 NIAH 成功: {'✅' if success else '❌'}")
    print(f"📝 回答: {result}")

    format_metrics(metrics)


def main():
    """メインデモ"""
    print("🚀 Fast-dLLM × LongLLaDA クイックスタート（修正版）")
    print("=" * 50)
    print("統合された拡散言語モデルの高速推論＆長文対応デモ")
    print("✨ Fast-dLLMの正しいデュアルキャッシュ実装を使用")
    print()

    # 環境チェック
    check_environment()

    try:
        # 1. 基本生成デモ
        model, tokenizer = demo_basic_generation()

        # 2. 長文コンテキストデモ
        demo_long_context(model, tokenizer)

        # 3. キャッシュ方式比較
        demo_cache_comparison()

        # 4. 速度比較デモ
        demo_speed_comparison()

        # 5. 簡単NIAHテスト
        demo_niah_simple()

        print("\n🎉 クイックスタート完了！")
        print("=" * 50)
        print("🔍 実装確認:")
        print("  ✅ Fast-dLLM デュアルキャッシュ実装")
        print("  ✅ LongLLaDA RoPE スケーリング")
        print("  ✅ 信頼度ベース並列デコーディング")
        print("  ✅ 統合された高速長文生成")
        print()
        print("📋 次のステップ:")
        print("  📊 詳細評価: python run_integrated_experiment.py")
        print("  🧑‍⚖️ LLM Judge: python llm_judge_evaluation.py")
        print("  📚 カスタマイズ: integrated_generation.py を編集")

    except Exception as e:
        print(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()

        print("\n🔧 トラブルシューティング:")
        print("  1. GPU メモリ不足 → block_length を大きく")
        print("  2. モデル読み込みエラー → インターネット接続確認")
        print("  3. パッケージエラー → pip install -r requirements.txt")
        print("  4. インポートエラー → Fast-dLLM/llada パスの確認")


if __name__ == "__main__":
    main()
