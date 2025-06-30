#!/usr/bin/env python3
"""
Fast-dLLM × LongLLaDA メモリ使用量テストスクリプト
CUDA OOM エラーの原因調査と最適化検証用
"""

import torch
import time
import gc
from integrated_generation import (
    load_model_with_scaling,
    safe_generate_with_fallback,
    get_memory_usage,
    optimize_parameters_for_memory
)


def monitor_memory(description=""):
    """メモリ使用量を監視・表示"""
    if torch.cuda.is_available():
        memory_info = get_memory_usage()
        print(f"💾 {description}")
        print(f"   使用済み: {memory_info['allocated']:.2f}GB")
        print(f"   予約済み: {memory_info['reserved']:.2f}GB")
        print(f"   空き容量: {memory_info['free']:.2f}GB")
        print(f"   総容量: {memory_info['total']:.2f}GB")
        return memory_info
    return None


def test_memory_optimization():
    """メモリ最適化のテスト"""
    print("🧪 メモリ最適化テスト開始")
    print("=" * 50)

    # 初期メモリ状態
    monitor_memory("初期状態")

    # モデル読み込み前のメモリ解放
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print("\n📥 モデル読み込みテスト...")

    try:
        # 軽量モデルテスト
        monitor_memory("モデル読み込み前")

        model, tokenizer, config = load_model_with_scaling(
            'GSAI-ML/LLaDA-8B-Instruct',
            scaling_factor=1  # まず標準設定でテスト
        )

        monitor_memory("モデル読み込み後")

        # 複数の入力長でテスト
        test_inputs = [
            ("短文テスト", "こんにちは！"),
            ("中文テスト", "プログラミングについて詳しく説明してください。" * 50),
            ("長文テスト", "AIの歴史と未来について詳しく論述してください。" * 200)
        ]

        for test_name, prompt_text in test_inputs:
            print(f"\n🔬 {test_name}")
            print("-" * 30)

            # プロンプト準備
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer(
                formatted_prompt, return_tensors='pt').input_ids.to(model.device)

            print(f"📏 入力長: {input_ids.shape[1]} トークン")

            # 最適化パラメータ計算
            memory_info = get_memory_usage()
            available_memory = memory_info['free'] if memory_info else 10
            optimized_params = optimize_parameters_for_memory(
                input_ids.shape[1],
                available_memory
            )
            print(f"🔧 推奨パラメータ: {optimized_params}")

            monitor_memory("生成前")

            try:
                # 安全生成でテスト
                start_time = time.time()
                outputs, nfe, metrics = safe_generate_with_fallback(
                    model=model,
                    prompt=input_ids,
                    temperature=0.0
                )
                end_time = time.time()

                result = tokenizer.decode(
                    outputs[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                print(f"✅ 生成成功!")
                print(f"   ⏱️  時間: {end_time - start_time:.2f}秒")
                print(f"   🚀 速度: {metrics['tokens_per_second']:.1f} tok/s")
                print(f"   📝 結果: {result[:100]}...")

                monitor_memory("生成後")

            except Exception as e:
                print(f"❌ エラー: {e}")
                monitor_memory("エラー後")

                # メモリクリーンアップ
                torch.cuda.empty_cache()
                gc.collect()

            print()

    except Exception as e:
        print(f"❌ 重大なエラー: {e}")
        monitor_memory("重大エラー後")

    finally:
        # 最終クリーンアップ
        print("🧹 メモリクリーンアップ...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        monitor_memory("クリーンアップ後")


def test_progressive_memory_usage():
    """段階的メモリ使用量テスト"""
    print("\n📈 段階的メモリ使用量テスト")
    print("=" * 50)

    # 異なるコンテキスト長でのメモリ使用パターンを調査
    context_lengths = [100, 500, 1000, 2000, 4000]

    base_text = "これはメモリテスト用のサンプルテキストです。"

    for length in context_lengths:
        print(f"\n🔍 {length} トークンテスト")

        # テキスト生成
        repeated_text = (
            # 概算
            base_text * (length // len(base_text.split()) + 1))[:length*5]

        # 推奨パラメータ表示
        params = optimize_parameters_for_memory(length)
        print(f"   推奨設定: {params}")

        # 予想メモリ使用量の計算（概算）
        vocab_size = 32000  # LLaDAの語彙サイズ
        model_precision = 2  # float16 = 2 bytes
        estimated_memory_gb = (length * vocab_size *
                               model_precision) / (1024**3)

        print(f"   予想メモリ: {estimated_memory_gb:.2f}GB (logitsのみ)")

        if estimated_memory_gb > 5:
            print("   ⚠️  大きなメモリ使用量が予想されます")


if __name__ == "__main__":
    print("🚀 Fast-dLLM メモリ最適化テスト")
    print("=" * 50)

    # CUDA利用可能性チェック
    if not torch.cuda.is_available():
        print("❌ CUDA が利用できません。CPUモードではメモリテストを実行できません。")
        exit(1)

    # GPU情報表示
    print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
    print(
        f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print()

    # テスト実行
    test_progressive_memory_usage()
    test_memory_optimization()

    print("\n✅ メモリテスト完了!")
    print("🔧 問題が発生した場合は、block_length を大きくするか steps を小さくしてください。")
