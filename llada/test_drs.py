import torch
import numpy as np
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate, generate_with_drs_fixed


def test_drs_basic():
    """DRSの基本機能テスト"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")

    try:
        # モデルをロード
        print("モデルをロード中...")
        model = LLaDAModelLM.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True
        )
        print("モデルロード完了")

        # よりシンプルで確実に回答できるプロンプト
        prompt = "What are the benefits of exercise for human health? Please list at least 5 benefits and explain each one briefly."
        m = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_formatted)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # 改善されたテストパラメータ
        gen_length = 128
        block_length = 32
        steps = 96

        print(f"\nテスト設定:")
        print(f"  生成長: {gen_length}")
        print(f"  ブロック長: {block_length}")
        print(f"  総ステップ: {steps}")
        print(f"  プロンプト: {prompt}")

        print("\n" + "="*50)
        print("ベースライン生成")
        print("="*50)

        # ベースライン生成
        baseline_out, baseline_nfe = generate(
            model, input_ids, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=0., remasking='low_confidence'
        )

        print("\n" + "="*50)
        print("DRS生成")
        print("="*50)

        # DRS生成（修正版） - より大きなt_baseと緩い閾値
        drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
            model, input_ids, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=0., threshold=0.7, t_base=12
        )

        # 結果を比較
        print("\n" + "="*50)
        print("結果比較")
        print("="*50)
        print(f"ベースライン NFE: {baseline_nfe}")
        print(f"DRS NFE: {drs_nfe}")
        nfe_reduction = ((baseline_nfe - drs_nfe) /
                         baseline_nfe) * 100 if baseline_nfe > 0 else 0
        print(f"NFE削減率: {nfe_reduction:.1f}%")
        print(f"ブロック曖昧度スコア: {ambiguity_scores}")

        # 出力をデコード
        baseline_text = tokenizer.batch_decode(
            baseline_out[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        drs_text = tokenizer.batch_decode(
            drs_out[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        print(f"\nベースライン出力:")
        print(f"'{baseline_text}'")
        print(f"\nDRS出力:")
        print(f"'{drs_text}'")

        # 品質評価（簡単な指標）
        baseline_length = len(baseline_text.split())
        drs_length = len(drs_text.split())
        print(f"\n品質指標:")
        print(f"  ベースライン単語数: {baseline_length}")
        print(f"  DRS単語数: {drs_length}")

        return baseline_nfe, drs_nfe, ambiguity_scores

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None, None, None


def test_drs_multiple_prompts():
    """複数のプロンプトでDRSをテスト"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # よりシンプルで確実に回答できるプロンプト
    prompts = [
        "Explain the water cycle in simple terms. Include evaporation, condensation, and precipitation.",
        "What are the main differences between cats and dogs as pets? List 3 key differences.",
        "Solve this math problem: If a book costs $15 and you buy 3 books, how much do you pay in total?",
        "Describe the four seasons and what happens in each one."
    ]

    try:
        # モデルをロード
        model = LLaDAModelLM.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True
        )

        results = []

        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"テスト {i+1}: {prompt}")
            print(f"{'='*60}")

            # プロンプトを準備
            m = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt_formatted)['input_ids']
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            # DRS生成（修正版） - 改善されたパラメータ
            drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
                model, input_ids, steps=96, gen_length=128,
                block_length=32, temperature=0., threshold=0.7, t_base=12
            )

            # 結果を保存
            drs_text = tokenizer.batch_decode(
                drs_out[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0]

            results.append({
                'prompt': prompt,
                'nfe': drs_nfe,
                'ambiguity_scores': ambiguity_scores,
                'output': drs_text
            })

            print(f"NFE: {drs_nfe}")
            print(f"曖昧度スコア: {ambiguity_scores}")
            print(f"出力: '{drs_text[:100]}...'")

        return results

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None


def test_drs_ablation():
    """DRSのアブレーションスタディ"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # モデルをロード
        model = LLaDAModelLM.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True
        )

        # よりシンプルなテストプロンプト
        prompt = "Write a short story about a cat who discovers a hidden treasure in the garden. Make it interesting and fun to read."
        m = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_formatted)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # 改善されたパラメータ設定
        test_configs = [
            {'t_base': 8, 'threshold': 0.6, 'name': '低ベース・低閾値'},
            {'t_base': 12, 'threshold': 0.7, 'name': '中ベース・中閾値'},
            {'t_base': 16, 'threshold': 0.8, 'name': '高ベース・高閾値'},
        ]

        print(f"\nアブレーションスタディ - プロンプト: {prompt}")
        print("="*70)

        for config in test_configs:
            print(
                f"\n設定: {config['name']} (t_base={config['t_base']}, threshold={config['threshold']})")
            print("-" * 50)

            # DRS生成（修正版）
            drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
                model, input_ids, steps=96, gen_length=128,
                block_length=32, temperature=0.,
                threshold=config['threshold'], t_base=config['t_base']
            )

            drs_text = tokenizer.batch_decode(
                drs_out[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0]

            print(f"NFE: {drs_nfe}")
            print(f"曖昧度スコア: {ambiguity_scores}")
            print(f"出力: '{drs_text[:150]}...'")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    print("DRS (Dynamic Refinement Scheduling) テスト開始")
    print("=" * 60)

    # 基本テスト
    print("\n1. 基本機能テスト")
    test_drs_basic()

    # 複数プロンプトテスト
    print("\n2. 複数プロンプトテスト")
    test_drs_multiple_prompts()

    # アブレーションスタディ
    print("\n3. アブレーションスタディ")
    test_drs_ablation()

    print("\n" + "=" * 60)
    print("全テスト完了")
