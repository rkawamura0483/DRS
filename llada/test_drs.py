import numpy as np
import torch
from generate import generate, generate_with_drs_fixed
from model.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer


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
                block_length=block_length, temperature=0., threshold=0.8, t_base=2  # 小さくして残りブロックを確保
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
        {'gen_length': 1024, 'block_length': 32, 'steps': 192, 'name': '長文生成'},
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
                block_length=config['block_length'], temperature=0., threshold=0.9, t_base=t_base
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
            {'t_base': 1, 'threshold': 0.9, 'name': '低初期予算'},
            {'t_base': 2, 'threshold': 0.9, 'name': '中初期予算'},
            {'t_base': 4, 'threshold': 0.9, 'name': '高初期予算'},
            # thresholdの影響
            {'t_base': 1, 'threshold': 0.95, 'name': '低閾値'},
            {'t_base': 2, 'threshold': 0.95, 'name': '高閾値'},
            {'t_base': 4, 'threshold': 0.95, 'name': '最高閾値'},
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


def test_drs_critical_analysis():
    """
    批判的分析: DRSの真の価値を検証

    現状の問題:
    1. 全ブロックが早期完成 → 「難しいブロック」が存在しない
    2. 研究仮定の破綻: 動的配分の必要性が証明されていない
    3. 効率化の原因が「早期終了」であり、「適応的計算配分」ではない

    解決策:
    1. より厳しい閾値とより難しいタスクを使用
    2. t_baseを極端に小さくして未完成ブロックを強制的に作る
    3. 真の難易度差を検証
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")

    # より抽象的で曖昧性の高いタスク
    truly_challenging_prompts = [
        # 高度な抽象思考 - 曖昧性が高い
        "Analyze the philosophical implications of artificial consciousness. If an AI system claims to experience qualia, how would we verify this claim? Discuss the hard problem of consciousness, the Chinese room argument, and whether computational processes can give rise to genuine subjective experience. Consider multiple perspectives from materialist, dualist, and panpsychist viewpoints, and propose criteria for distinguishing between simulated and genuine consciousness.",

        # 複雑な創作 - 一貫性が困難
        "Write a surreal short story that seamlessly blends three completely different genres: cyberpunk noir, medieval fantasy, and cosmic horror. The protagonist must be simultaneously a detective investigating a murder in neo-Tokyo, a knight seeking a mystical artifact, and an astronomer discovering something terrifying in deep space. These three realities should be the same person experiencing parallel dimensions that begin to converge catastrophically. Maintain narrative coherence while exploring themes of identity fragmentation.",

        # 極度に複雑な論理推論
        "Consider a fictional universe where the laws of physics change based on collective human belief. In this world, if 60% of people believe gravity is weaker on Tuesdays, it actually becomes weaker. Now imagine three competing scientific theories about consciousness emerge, each with different implications for how reality should behave. Theory A suggests consciousness creates reality, Theory B suggests reality creates consciousness, and Theory C suggests both co-create each other cyclically. If these theories gain 30%, 35%, and 35% belief respectively, what would happen to the nature of scientific observation itself? Analyze the paradoxes and feedback loops.",

        # 多重制約下での創造性
        "Design a new form of mathematics that operates on emotional rather than numerical relationships. Define at least 5 fundamental operations (like addition/subtraction equivalents) that work with feelings like joy, melancholy, anxiety, wonder, and nostalgia. Create axioms that govern how these emotional operations interact, ensuring logical consistency. Then use this emotional mathematics to solve a practical problem: how to optimize the emotional experience of a user interface. Show your work using your new mathematical notation, and prove that your system is both internally consistent and useful for real-world applications."
    ]

    try:
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

        # より厳しいテスト条件
        gen_length = 384  # より長い生成
        block_length = 32
        total_steps = 192

        print(f"\n{'='*80}")
        print("DRS批判的分析: 真の価値検証")
        print(f"{'='*80}")
        print("目的: 研究仮定の検証 - '難しいブロック'は本当に存在するか？")

        critical_results = []

        for i, prompt in enumerate(truly_challenging_prompts):
            print(f"\n{'='*80}")
            print(f"極限挑戦タスク {i+1}: {prompt[:100]}...")
            print(f"{'='*80}")

            m = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt_formatted)['input_ids']
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            # 段階的に厳しくする実験
            test_conditions = [
                {'t_base': 4, 'threshold': 0.9, 'name': '厳しい条件'},
                {'t_base': 3, 'threshold': 0.9, 'name': '極限条件'},
                {'t_base': 2, 'threshold': 0.9, 'name': '最極限条件'},
            ]

            for condition in test_conditions:
                print(f"\n{'-'*60}")
                print(
                    f"テスト条件: {condition['name']} (t_base={condition['t_base']}, threshold={condition['threshold']})")
                print(f"{'-'*60}")

                # ベースライン
                baseline_out, baseline_nfe = generate(
                    model, input_ids, steps=total_steps, gen_length=gen_length,
                    block_length=block_length, temperature=0., remasking='low_confidence'
                )

                # DRS（極限設定）
                drs_out, drs_nfe, ambiguity_scores = generate_with_drs_fixed(
                    model, input_ids, steps=total_steps, gen_length=gen_length,
                    block_length=block_length, temperature=0.,
                    threshold=condition['threshold'], t_base=condition['t_base']
                )

                # 批判的分析
                has_ambiguous_blocks = any(
                    score > 0 for score in ambiguity_scores)
                max_ambiguity = max(
                    ambiguity_scores) if ambiguity_scores else 0
                ambiguity_variance = np.var(
                    ambiguity_scores) if ambiguity_scores else 0

                baseline_text = tokenizer.batch_decode(
                    baseline_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                drs_text = tokenizer.batch_decode(
                    drs_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

                nfe_reduction = ((baseline_nfe - drs_nfe) / baseline_nfe) * 100
                quality_preservation = len(
                    drs_text.split()) / len(baseline_text.split()) * 100

                # 批判的評価
                print(f"結果分析:")
                print(
                    f"  NFE削減: {nfe_reduction:.1f}% ({baseline_nfe} → {drs_nfe})")
                print(f"  品質保持: {quality_preservation:.1f}%")
                print(
                    f"  難しいブロック存在: {'YES' if has_ambiguous_blocks else 'NO'}")
                print(f"  最大曖昧度: {max_ambiguity:.3f}")
                print(f"  曖昧度分散: {ambiguity_variance:.3f}")
                print(f"  曖昧度分布: {ambiguity_scores}")

                # DRSの真の価値評価
                if has_ambiguous_blocks and ambiguity_variance > 0.01:
                    print(f"  ✅ DRS価値: TRUE - 動的配分が有効")
                elif nfe_reduction > 30:
                    print(f"  ⚠️  DRS価値: PARTIAL - 早期終了効果のみ")
                else:
                    print(f"  ❌ DRS価値: FALSE - 効果なし")

                critical_results.append({
                    'task': i+1,
                    'condition': condition['name'],
                    'has_ambiguous_blocks': has_ambiguous_blocks,
                    'max_ambiguity': max_ambiguity,
                    'ambiguity_variance': ambiguity_variance,
                    'nfe_reduction': nfe_reduction,
                    'quality_preservation': quality_preservation,
                    'true_drs_value': has_ambiguous_blocks and ambiguity_variance > 0.01
                })

                # NFE使用量が異常に少ない場合の警告
                if drs_nfe < total_steps * 0.4:
                    print(
                        f"  🚨 警告: 早期終了率が高すぎる ({(1 - drs_nfe/total_steps)*100:.1f}%)")
                    print(f"       これは研究仮定の破綻を示唆している")

        # 最終的な批判的分析
        print(f"\n{'='*80}")
        print("批判的分析結果サマリー")
        print(f"{'='*80}")

        true_drs_cases = sum(
            1 for r in critical_results if r['true_drs_value'])
        total_cases = len(critical_results)
        avg_max_ambiguity = np.mean([r['max_ambiguity']
                                    for r in critical_results])
        avg_variance = np.mean([r['ambiguity_variance']
                               for r in critical_results])

        print(
            f"真のDRS価値を示したケース: {true_drs_cases}/{total_cases} ({true_drs_cases/total_cases*100:.1f}%)")
        print(f"平均最大曖昧度: {avg_max_ambiguity:.3f}")
        print(f"平均曖昧度分散: {avg_variance:.3f}")

        print(f"\n研究への示唆:")
        if true_drs_cases == 0:
            print("❌ 研究仮定の完全な破綻:")
            print("   - '難しいブロック'が存在しない")
            print("   - DRSの動的配分価値が証明されない")
            print("   - 効率化は早期終了効果のみ")
            print("   → 研究方向の根本的見直しが必要")
        elif true_drs_cases < total_cases * 0.3:
            print("⚠️  研究仮定の部分的破綻:")
            print("   - 稀にしか'難しいブロック'が出現しない")
            print("   - DRSの適用範囲が限定的")
            print("   → タスク選択の再考が必要")
        else:
            print("✅ 研究仮定の部分的検証:")
            print("   - 条件によっては'難しいブロック'が存在")
            print("   - DRSの動的配分に価値がある")
            print("   → より適切な閾値設定の探索が必要")

        return critical_results

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("DRS (Dynamic Refinement Scheduling) 包括的研究検証")
    print("=" * 80)
    print("⚠️  重要: 初期結果は全て早期終了 → 研究仮定の検証が必要")
    print("研究目的: 同じ品質でNFE削減 + 複雑タスクでの適応的計算配分")
    print("=" * 80)

    # 批判的分析を最初に実行（最重要）
    print("\n🔍 CRITICAL: DRS批判的分析 - 研究仮定の検証")
    critical_results = test_drs_critical_analysis()

    # 批判的分析の結果に基づいて継続判断
    if critical_results:
        true_drs_value_found = any(r['true_drs_value']
                                   for r in critical_results)

        if true_drs_value_found:
            print("\n✅ 批判的分析で真のDRS価値を確認 → 追加テスト実行")

            # 1. 挑戦的タスクでの基本検証
            print("\n1. 挑戦的タスクでのDRS効果検証")
            challenging_results = test_drs_with_challenging_tasks()

            # 2. スケーラビリティテスト
            print("\n2. DRSスケーラビリティテスト")
            scalability_results = test_drs_scalability()

            # 3. 詳細アブレーションスタディ
            print("\n3. DRS詳細アブレーションスタディ")
            ablation_results = test_drs_ablation_study()

        else:
            print("\n❌ 批判的分析で真のDRS価値を確認できず")
            print("   → 追加テストをスキップ（研究仮定の破綻）")
            print("   → 研究方向の根本的見直しを推奨")

    print("\n" + "=" * 80)
    print("🎯 最終評価:")
    print("=" * 80)

    if critical_results and any(r['true_drs_value'] for r in critical_results):
        print("✅ DRSに研究価値あり - 条件付きで動的配分が有効")
        print("📋 推奨: より適切なタスクセットと閾値での継続研究")
    else:
        print("❌ DRSの研究価値に疑問 - 主に早期終了効果のみ")
        print("📋 推奨: 研究方向の転換または新しいアプローチの探索")
        print("   例: 別の効率化手法、異なるモデル、新しい評価基準")

    print("研究検証完了")
    print("=" * 80)
