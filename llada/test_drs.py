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

        # より困難なテストプロンプト - 長い推論が必要
        prompt = "Solve this step by step: A company has three departments. Department A has 45 employees, Department B has 38 employees, and Department C has 52 employees. If the company wants to reorganize into 4 equal departments, how many employees should each new department have? Show all calculations and explain your reasoning in detail."
        m = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_formatted)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # より困難なテストパラメータ
        gen_length = 128  # より長い生成
        block_length = 32  # より大きなブロック
        steps = 96        # より多くのステップ

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

        # DRS生成（修正版） - より厳しい閾値とより少ないベースステップ
        drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
            model, input_ids, steps=steps, gen_length=gen_length,
            block_length=block_length, temperature=0., threshold=0.9, t_base=4
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

    prompts = [
        "Explain the process of photosynthesis in detail, including the light-dependent and light-independent reactions, their locations in the chloroplast, and the overall significance to life on Earth.",
        "Write a comprehensive analysis of the economic impacts of artificial intelligence on different sectors of the economy, including both positive and negative effects.",
        "Solve this complex math problem: A rectangular garden is 3 times as long as it is wide. If the perimeter is 80 meters, what are the dimensions? Then calculate the area and explain how you could divide it into 6 equal sections.",
        "Describe the complete water cycle, including all major processes like evaporation, condensation, precipitation, infiltration, and runoff, and explain how human activities affect each stage."
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
            print(f"テスト {i+1}: {prompt[:50]}...")
            print(f"{'='*60}")

            # プロンプトを準備
            m = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt_formatted)['input_ids']
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            # DRS生成（修正版） - より困難な条件
            drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
                model, input_ids, steps=96, gen_length=128,
                block_length=32, temperature=0., threshold=0.9, t_base=4
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

        # より複雑なテストプロンプト
        prompt = "Analyze the following business scenario: A startup company needs to decide between three different strategies for market entry. Strategy A requires an initial investment of $500,000 with projected monthly profits of $45,000 starting from month 6. Strategy B requires $300,000 initial investment with $25,000 monthly profits starting from month 3. Strategy C requires $800,000 with $60,000 monthly profits starting from month 8. Calculate the break-even point for each strategy, analyze the risks and benefits, and recommend the best approach with detailed reasoning."
        m = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_formatted)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # 異なるパラメータでテスト - より困難な条件
        test_configs = [
            {'t_base': 2, 'threshold': 0.7, 'name': '超低ベース・低閾値'},
            {'t_base': 4, 'threshold': 0.8, 'name': '低ベース・中閾値'},
            {'t_base': 6, 'threshold': 0.9, 'name': '中ベース・高閾値'},
        ]

        print(f"\nアブレーションスタディ - プロンプト: {prompt[:100]}...")
        print("="*70)

        for config in test_configs:
            print(
                f"\n設定: {config['name']} (t_base={config['t_base']}, threshold={config['threshold']})")
            print("-" * 50)

            # DRS生成（修正版） - より困難な条件
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
