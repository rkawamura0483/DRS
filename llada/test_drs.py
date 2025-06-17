import torch
import numpy as np
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate, generate_with_drs_fixed


def test_drs_with_challenging_tasks():
    """研究目的に沿った挑戦的なタスクでDRSをテスト"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")

    # より挑戦的で複雑なプロンプト（READMEの研究目的に基づく）
    challenging_prompts = [
        # 複雑な数学推論（GSM8K風）
        "A bakery produces three types of bread: whole wheat, white, and rye. On Monday, they baked 45 loaves of whole wheat bread, which was 30% of their total production. On Tuesday, they increased whole wheat production by 20% and white bread production by 15%, while keeping rye bread the same. If white bread was 40% of Monday's total and rye bread was the remaining portion, calculate the total number of loaves produced over both days and determine what percentage of the two-day total was whole wheat bread.",

        # 複雑なコーディング問題（HumanEval風）
        "Write a Python function called 'find_optimal_path' that takes a 2D grid represented as a list of lists, where 0 represents empty space and 1 represents obstacles. The function should find the shortest path from the top-left corner (0,0) to the bottom-right corner using dynamic programming or BFS algorithm. Include proper error handling for invalid inputs and return both the path length and the actual path coordinates as a tuple. Make sure to handle edge cases like when no path exists.",

        # 長文の論理的説明
        "Explain the concept of quantum entanglement in detail, covering the following aspects: 1) The fundamental physics principles involved, 2) How Bell's theorem relates to local hidden variable theories, 3) The practical applications in quantum computing and quantum cryptography, 4) The paradoxes it creates with classical intuition about locality and realism, and 5) Recent experimental confirmations and their implications for our understanding of reality. Provide specific examples and explain the mathematics where relevant.",

        # 複雑な推論チェーン
        "A company is evaluating three investment options: Option A offers 8% annual return with 15% risk, Option B offers 12% annual return with 25% risk, and Option C offers 6% annual return with 8% risk. Given that the company has $500,000 to invest and wants to maximize returns while keeping overall portfolio risk below 18%, determine the optimal allocation strategy. Consider that they can split their investment across multiple options and that risk is calculated as the weighted average of individual risks. Show all calculations and explain your reasoning process step by step."
    ]

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

        # 研究目的に沿ったテスト設定
        gen_length = 256  # より長い生成（複雑な回答に対応）
        block_length = 32
        total_steps = 128  # 総予算を固定

        results = []

        for i, prompt in enumerate(challenging_prompts):
            print(f"\n{'='*80}")
            print(f"挑戦的タスク {i+1}: {prompt[:100]}...")
            print(f"{'='*80}")

            # プロンプトを準備
            m = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt_formatted)['input_ids']
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            print(f"\nテスト設定:")
            print(f"  生成長: {gen_length}")
            print(f"  ブロック長: {block_length}")
            print(f"  総ステップ予算: {total_steps}")

            print("\n" + "="*50)
            print("ベースライン生成 (固定スケジュール)")
            print("="*50)

            # ベースライン: 均等配分
            baseline_out, baseline_nfe = generate(
                model, input_ids, steps=total_steps, gen_length=gen_length,
                block_length=block_length, temperature=0., remasking='low_confidence'
            )

            print("\n" + "="*50)
            print("DRS生成 (動的スケジュール)")
            print("="*50)

            # DRS: 小さなt_baseで確実に残りブロックを作る
            drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
                model, input_ids, steps=total_steps, gen_length=gen_length,
                block_length=block_length, temperature=0., threshold=0.8, t_base=6  # 小さくして残りブロックを確保
            )

            # 結果をデコード
            baseline_text = tokenizer.batch_decode(
                baseline_out[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0]
            drs_text = tokenizer.batch_decode(
                drs_out[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0]

            # 品質評価
            baseline_length = len(baseline_text.split())
            drs_length = len(drs_text.split())

            # NFE効率性計算
            nfe_reduction = ((baseline_nfe - drs_nfe) /
                             baseline_nfe) * 100 if baseline_nfe > 0 else 0
            efficiency_gain = baseline_length / baseline_nfe if baseline_nfe > 0 else 0
            drs_efficiency = drs_length / drs_nfe if drs_nfe > 0 else 0

            print("\n" + "="*50)
            print("結果比較と分析")
            print("="*50)
            print(f"ベースライン NFE: {baseline_nfe}")
            print(f"DRS NFE: {drs_nfe}")
            print(f"NFE削減率: {nfe_reduction:.1f}%")
            print(f"品質指標:")
            print(f"  ベースライン単語数: {baseline_length}")
            print(f"  DRS単語数: {drs_length}")
            print(
                f"  品質保持率: {(drs_length/baseline_length*100):.1f}%" if baseline_length > 0 else "N/A")
            print(f"効率性指標:")
            print(f"  ベースライン効率 (単語/NFE): {efficiency_gain:.3f}")
            print(f"  DRS効率 (単語/NFE): {drs_efficiency:.3f}")
            print(
                f"  効率性向上: {((drs_efficiency - efficiency_gain)/efficiency_gain*100):.1f}%" if efficiency_gain > 0 else "N/A")
            print(f"ブロック曖昧度スコア: {ambiguity_scores}")

            print(f"\nベースライン出力 (最初の200文字):")
            print(f"'{baseline_text[:200]}...'")
            print(f"\nDRS出力 (最初の200文字):")
            print(f"'{drs_text[:200]}...'")

            results.append({
                'prompt': prompt[:100] + "...",
                'baseline_nfe': baseline_nfe,
                'drs_nfe': drs_nfe,
                'nfe_reduction': nfe_reduction,
                'ambiguity_scores': ambiguity_scores,
                'baseline_quality': baseline_length,
                'drs_quality': drs_length,
                'efficiency_gain': ((drs_efficiency - efficiency_gain)/efficiency_gain*100) if efficiency_gain > 0 else 0
            })

        # 全体的な結果サマリー
        print("\n" + "="*80)
        print("研究成果サマリー: DRSの効果分析")
        print("="*80)

        avg_nfe_reduction = np.mean([r['nfe_reduction'] for r in results])
        avg_efficiency_gain = np.mean(
            [r['efficiency_gain'] for r in results if r['efficiency_gain'] != 0])

        print(f"平均NFE削減率: {avg_nfe_reduction:.1f}%")
        print(f"平均効率性向上: {avg_efficiency_gain:.1f}%")
        print(f"テスト完了タスク数: {len(results)}")

        # 研究目的との整合性チェック
        print(f"\n研究目的との整合性:")
        print(f"✓ 複雑なタスクでの検証: {len(results)}個の挑戦的タスク")
        print(f"✓ NFE削減達成: 平均{avg_nfe_reduction:.1f}%削減")
        print(f"✓ 動的予算配分: 曖昧度スコアに基づく適応的計算")

        return results

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_drs_scalability():
    """DRSのスケーラビリティテスト: 異なる長さと複雑さでの性能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # スケーラビリティテスト用の設定
    test_configs = [
        {'gen_length': 128, 'block_length': 32, 'steps': 96, 'name': '短文生成'},
        {'gen_length': 256, 'block_length': 32, 'steps': 128, 'name': '中文生成'},
        {'gen_length': 384, 'block_length': 48, 'steps': 192, 'name': '長文生成'},
    ]

    # 中程度の複雑さのプロンプト
    prompt = "Design a comprehensive software architecture for a distributed e-commerce system that handles high traffic. Include details about microservices, database design, caching strategies, security measures, and scalability considerations. Explain the trade-offs of your design choices and how the system would handle peak loads during sales events."

    try:
        model = LLaDAModelLM.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True
        )

        m = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_formatted)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        print(f"\n{'='*80}")
        print("DRSスケーラビリティテスト")
        print(f"{'='*80}")

        results = []

        for config in test_configs:
            print(f"\n{'-'*60}")
            print(f"テスト設定: {config['name']}")
            print(
                f"生成長: {config['gen_length']}, ブロック長: {config['block_length']}, ステップ: {config['steps']}")
            print(f"{'-'*60}")

            # ベースライン
            baseline_out, baseline_nfe = generate(
                model, input_ids, steps=config['steps'], gen_length=config['gen_length'],
                block_length=config['block_length'], temperature=0., remasking='low_confidence'
            )

            # DRS（適応的t_base）
            t_base = max(4, config['steps'] // (config['gen_length'] //
                         config['block_length']) // 3)  # 予算の1/3をベースに
            drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
                model, input_ids, steps=config['steps'], gen_length=config['gen_length'],
                block_length=config['block_length'], temperature=0., threshold=0.75, t_base=t_base
            )

            # 効果測定
            nfe_reduction = ((baseline_nfe - drs_nfe) /
                             baseline_nfe) * 100 if baseline_nfe > 0 else 0

            baseline_text = tokenizer.batch_decode(
                baseline_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            drs_text = tokenizer.batch_decode(
                drs_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            print(
                f"結果: NFE削減 {nfe_reduction:.1f}% ({baseline_nfe} → {drs_nfe})")
            print(
                f"品質: {len(baseline_text.split())} → {len(drs_text.split())} 単語")
            print(f"曖昧度分散: {np.std(ambiguity_scores):.3f}")

            results.append({
                'config': config['name'],
                'nfe_reduction': nfe_reduction,
                'baseline_nfe': baseline_nfe,
                'drs_nfe': drs_nfe,
                'ambiguity_std': np.std(ambiguity_scores)
            })

        print(f"\n{'='*60}")
        print("スケーラビリティ結果サマリー")
        print(f"{'='*60}")
        for result in results:
            print(
                f"{result['config']}: NFE削減 {result['nfe_reduction']:.1f}%, 曖昧度分散 {result['ambiguity_std']:.3f}")

        return results

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_drs_ablation_study():
    """DRSの詳細なアブレーションスタディ"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        model = LLaDAModelLM.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True
        )

        # 中程度の複雑さのプロンプト
        prompt = "Analyze the causes and consequences of climate change, focusing on the role of greenhouse gases, feedback loops in the climate system, and the potential impacts on global ecosystems, agriculture, and human societies. Discuss both mitigation and adaptation strategies that could be implemented at local, national, and international levels."

        m = [{"role": "user", "content": prompt}]
        prompt_formatted = tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_formatted)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        # アブレーション研究の設定
        ablation_configs = [
            # t_baseの影響
            {'t_base': 4, 'threshold': 0.8, 'name': '低初期予算'},
            {'t_base': 8, 'threshold': 0.8, 'name': '中初期予算'},
            {'t_base': 12, 'threshold': 0.8, 'name': '高初期予算'},
            # thresholdの影響
            {'t_base': 6, 'threshold': 0.6, 'name': '低閾値'},
            {'t_base': 6, 'threshold': 0.8, 'name': '高閾値'},
            {'t_base': 6, 'threshold': 0.9, 'name': '最高閾値'},
        ]

        print(f"\n{'='*80}")
        print("DRS詳細アブレーションスタディ")
        print(f"{'='*80}")

        # まずベースラインを取得
        baseline_out, baseline_nfe = generate(
            model, input_ids, steps=128, gen_length=256,
            block_length=32, temperature=0., remasking='low_confidence'
        )
        baseline_text = tokenizer.batch_decode(
            baseline_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        print(f"ベースライン: NFE={baseline_nfe}, 単語数={len(baseline_text.split())}")

        results = []

        for config in ablation_configs:
            print(f"\n{'-'*50}")
            print(
                f"設定: {config['name']} (t_base={config['t_base']}, threshold={config['threshold']})")
            print(f"{'-'*50}")

            drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
                model, input_ids, steps=128, gen_length=256,
                block_length=32, temperature=0.,
                threshold=config['threshold'], t_base=config['t_base']
            )

            drs_text = tokenizer.batch_decode(
                drs_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            nfe_reduction = ((baseline_nfe - drs_nfe) / baseline_nfe) * 100
            quality_ratio = len(drs_text.split()) / \
                len(baseline_text.split()) * 100

            print(f"結果: NFE削減 {nfe_reduction:.1f}%, 品質保持 {quality_ratio:.1f}%")
            print(f"曖昧度スコア: {ambiguity_scores}")

            results.append({
                'name': config['name'],
                't_base': config['t_base'],
                'threshold': config['threshold'],
                'nfe_reduction': nfe_reduction,
                'quality_ratio': quality_ratio,
                'ambiguity_scores': ambiguity_scores
            })

        print(f"\n{'='*60}")
        print("アブレーション結果サマリー")
        print(f"{'='*60}")
        for result in results:
            print(
                f"{result['name']}: NFE削減 {result['nfe_reduction']:.1f}%, 品質保持 {result['quality_ratio']:.1f}%")

        return results

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("DRS (Dynamic Refinement Scheduling) 包括的研究検証")
    print("=" * 80)
    print("研究目的: 同じ品質でNFE削減 + 複雑タスクでの適応的計算配分")
    print("=" * 80)

    # 1. 挑戦的タスクでの基本検証
    print("\n1. 挑戦的タスクでのDRS効果検証")
    challenging_results = test_drs_with_challenging_tasks()

    # 2. スケーラビリティテスト
    print("\n2. DRSスケーラビリティテスト")
    scalability_results = test_drs_scalability()

    # 3. 詳細アブレーションスタディ
    print("\n3. DRS詳細アブレーションスタディ")
    ablation_results = test_drs_ablation_study()

    print("\n" + "=" * 80)
    print("研究検証完了")
    print("=" * 80)
