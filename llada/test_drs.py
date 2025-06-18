# 大幅改善版DRS検証システム - 評価バイアス修正版
import logging
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from generate import generate_with_drs, generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedQualityEvaluator:
    """大幅改善された品質評価システム - バイアス修正版"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def evaluate_text_coherence(self, text):
        """テキスト一貫性の評価（独立指標）"""
        if not text or len(text) < 10:
            return 0.0

        # 文の構造評価
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        if len(valid_sentences) == 0:
            return 0.0

        # 平均文長の妥当性
        avg_sentence_length = np.mean(
            [len(s.split()) for s in valid_sentences])
        length_score = min(1.0, avg_sentence_length / 15)  # 適度な文長を評価

        # 語彙多様性
        words = text.lower().split()
        unique_words = set(words)
        diversity_score = len(unique_words) / max(len(words), 1)

        return 0.6 * length_score + 0.4 * diversity_score

    def evaluate_content_completeness(self, prompt, text):
        """プロンプトに対する回答完全性の評価"""
        prompt_lower = prompt.lower()
        text_lower = text.lower()

        if "rectangular" in prompt_lower and "garden" in prompt_lower:
            # 長方形の庭問題
            required_elements = ["area", "length", "width", "15", "8", "120"]
            found = sum(1 for elem in required_elements if elem in text_lower)
            return found / len(required_elements)

        elif "python" in prompt_lower and "function" in prompt_lower and "factorial" in prompt_lower:
            # 階乗関数問題
            required_elements = ["def", "factorial",
                                 "return", "recursion", "if", "else"]
            found = sum(1 for elem in required_elements if elem in text_lower)
            return found / len(required_elements)

        # 一般的な完全性評価
        words_in_prompt = set(prompt_lower.split())
        words_in_text = set(text_lower.split())
        coverage = len(words_in_prompt & words_in_text) / \
            max(len(words_in_prompt), 1)
        return min(1.0, coverage * 2)  # プロンプトキーワードカバレッジ

    def evaluate_mathematical_accuracy(self, prompt, text):
        """数学的正確性の評価"""
        if "rectangular" in prompt.lower() and "15" in prompt and "8" in prompt:
            # 正確な答え：120平方メートル
            if "120" in text:
                return 1.0
            elif any(calc in text.lower() for calc in ["15 * 8", "15*8", "8 * 15", "8*15"]):
                return 0.8  # 計算式は正しい
            elif "area" in text.lower() and ("length" in text.lower() or "width" in text.lower()):
                return 0.5  # 概念は理解している
            else:
                return 0.0
        return 0.5  # 数学問題でない場合は中立

    def comprehensive_quality_assessment(self, prompt, text):
        """包括的品質評価（独立指標）"""
        coherence = self.evaluate_text_coherence(text)
        completeness = self.evaluate_content_completeness(prompt, text)
        math_accuracy = self.evaluate_mathematical_accuracy(prompt, text)

        # 重み付き統合スコア
        final_score = 0.4 * coherence + 0.4 * completeness + 0.2 * math_accuracy

        return final_score, {
            'coherence': coherence,
            'completeness': completeness,
            'math_accuracy': math_accuracy
        }

    def evaluate_efficiency_quality_tradeoff(self, baseline_nfe, drs_nfe, baseline_quality, drs_quality):
        """効率性と品質のトレードオフ評価"""
        nfe_ratio = drs_nfe / max(baseline_nfe, 1)
        quality_gain = drs_quality - baseline_quality

        # 効率性スコア（NFE削減を評価）
        efficiency_score = max(0, 2 - nfe_ratio)  # NFE半減で最高評価

        # 品質向上スコア
        quality_improvement_score = max(0, quality_gain * 10)  # 0.1の改善で1点

        # 統合トレードオフスコア
        if quality_gain > 0.02:  # 2%以上の品質向上
            if nfe_ratio <= 1.1:  # NFE増加10%以内
                return "EXCELLENT"
            elif nfe_ratio <= 1.2:  # NFE増加20%以内
                return "GOOD"
            else:
                return "MODERATE"
        elif quality_gain > -0.02:  # 品質維持（±2%）
            if nfe_ratio <= 0.9:  # NFE削減10%以上
                return "GOOD"
            elif nfe_ratio <= 1.0:  # NFE維持
                return "MODERATE"
            else:
                return "POOR"
        else:  # 品質低下
            return "POOR"


def enhanced_drs_validation():
    """改善版DRS仮説検証システム"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔬 改善版DRS仮説検証開始 (デバイス: {device})")

    # より多様なテストプロンプト
    test_prompts = [
        {
            "text": "Calculate the area of a rectangular garden with length 15 meters and width 8 meters.",
            "type": "math",
            "expected_answer": "120"
        },
        {
            "text": "Write a Python function to find the factorial of a number using recursion.",
            "type": "code",
            "expected_elements": ["def", "factorial", "recursion"]
        },
        {
            "text": "Explain the concept of photosynthesis in plants.",
            "type": "explanation",
            "expected_elements": ["chlorophyll", "sunlight", "carbon dioxide"]
        }
    ]

    try:
        print("📦 モデルロード中...")
        model = LLaDAModelLM.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/LLaDA-8B-Instruct',
            trust_remote_code=True
        )
        model.tokenizer = tokenizer

        mask_id = tokenizer.mask_token_id or model.config.mask_token_id or 126336
        print(f"✅ モデルロード完了 (mask_id={mask_id})")

        evaluator = EnhancedQualityEvaluator(tokenizer)

        gen_length = 128
        block_length = 32
        total_steps = 128

        all_results = []

        for i, prompt_data in enumerate(test_prompts):
            prompt = prompt_data["text"]
            print(f"\n{'='*80}")
            print(f"📝 テストプロンプト {i+1} ({prompt_data['type']}): {prompt}")
            print(f"{'='*80}")

            m = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt_formatted)['input_ids']
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            # ベースライン生成
            print("\n🎯 ベースライン生成中...")
            baseline_out, baseline_nfe = generate_with_dual_cache(
                model, input_ids, steps=total_steps, gen_length=gen_length,
                block_length=block_length, temperature=0., remasking='low_confidence',
                mask_id=mask_id
            )
            baseline_text = tokenizer.batch_decode(
                baseline_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            baseline_quality, baseline_details = evaluator.comprehensive_quality_assessment(
                prompt, baseline_text)

            # DRS設定をより幅広くテスト
            drs_configs = [
                {'t_base': 4, 'threshold': 0.7, 'name': 'DRS-Aggressive'},
                {'t_base': 8, 'threshold': 0.8, 'name': 'DRS-Balanced'},
                {'t_base': 12, 'threshold': 0.9, 'name': 'DRS-Conservative'},
            ]

            for config in drs_configs:
                print(f"\n{'-'*60}")
                print(
                    f"🧪 {config['name']} (t_base={config['t_base']}, threshold={config['threshold']})")
                print(f"{'-'*60}")

                # DRS生成
                drs_out, drs_nfe, ambiguity_scores = generate_with_drs(
                    model, input_ids, steps=total_steps, gen_length=gen_length,
                    block_length=block_length, temperature=0.,
                    threshold=config['threshold'], t_base=config['t_base'],
                    mask_id=mask_id
                )
                drs_text = tokenizer.batch_decode(
                    drs_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

                # 独立品質評価
                drs_quality, drs_details = evaluator.comprehensive_quality_assessment(
                    prompt, drs_text)

                # トレードオフ評価
                tradeoff_rating = evaluator.evaluate_efficiency_quality_tradeoff(
                    baseline_nfe, drs_nfe, baseline_quality, drs_quality
                )

                # 詳細出力
                print(f"\n📊 結果比較:")
                print(f"  🎯 ベースライン:")
                print(f"     NFE: {baseline_nfe}")
                print(
                    f"     品質: {baseline_quality:.3f} (coherence:{baseline_details['coherence']:.3f}, completeness:{baseline_details['completeness']:.3f}, math:{baseline_details['math_accuracy']:.3f})")
                print(f"     テキスト: {baseline_text[:150]}...")

                print(f"  ⚡ {config['name']}:")
                print(
                    f"     NFE: {drs_nfe} (差分: {drs_nfe-baseline_nfe:+d}, 比率: {drs_nfe/baseline_nfe:.2f}x)")
                print(
                    f"     品質: {drs_quality:.3f} (coherence:{drs_details['coherence']:.3f}, completeness:{drs_details['completeness']:.3f}, math:{drs_details['math_accuracy']:.3f})")
                print(f"     品質差分: {drs_quality-baseline_quality:+.3f}")
                print(f"     テキスト: {drs_text[:150]}...")
                print(f"     トレードオフ評価: {tradeoff_rating}")
                print(f"     曖昧度スコア: {[f'{s:.3f}' for s in ambiguity_scores]}")

                # 結果記録
                all_results.append({
                    'prompt_id': i+1,
                    'prompt_type': prompt_data['type'],
                    'config': config['name'],
                    'baseline_nfe': baseline_nfe,
                    'drs_nfe': drs_nfe,
                    'nfe_ratio': drs_nfe / baseline_nfe,
                    'baseline_quality': baseline_quality,
                    'drs_quality': drs_quality,
                    'quality_gain': drs_quality - baseline_quality,
                    'tradeoff_rating': tradeoff_rating,
                    'avg_ambiguity': np.mean(ambiguity_scores)
                })

        # 統合分析
        print(f"\n{'='*80}")
        print("🎯 DRS仮説検証結果 - 統合分析")
        print(f"{'='*80}")

        excellent_cases = sum(
            1 for r in all_results if r['tradeoff_rating'] == 'EXCELLENT')
        good_cases = sum(
            1 for r in all_results if r['tradeoff_rating'] == 'GOOD')
        moderate_cases = sum(
            1 for r in all_results if r['tradeoff_rating'] == 'MODERATE')
        poor_cases = sum(
            1 for r in all_results if r['tradeoff_rating'] == 'POOR')
        total_cases = len(all_results)

        avg_nfe_ratio = np.mean([r['nfe_ratio'] for r in all_results])
        avg_quality_gain = np.mean([r['quality_gain'] for r in all_results])
        avg_ambiguity = np.mean([r['avg_ambiguity'] for r in all_results])

        print(f"📊 性能分布:")
        print(
            f"  🌟 EXCELLENT: {excellent_cases}/{total_cases} ({100*excellent_cases/total_cases:.1f}%)")
        print(
            f"  ✅ GOOD: {good_cases}/{total_cases} ({100*good_cases/total_cases:.1f}%)")
        print(
            f"  ⚠️ MODERATE: {moderate_cases}/{total_cases} ({100*moderate_cases/total_cases:.1f}%)")
        print(
            f"  ❌ POOR: {poor_cases}/{total_cases} ({100*poor_cases/total_cases:.1f}%)")

        print(f"\n📈 平均指標:")
        print(f"  NFE比率: {avg_nfe_ratio:.3f}x")
        print(f"  品質向上: {avg_quality_gain:+.3f}")
        print(f"  平均曖昧度: {avg_ambiguity:.3f}")

        # DRS仮説の最終判定
        success_rate = (excellent_cases + good_cases) / total_cases

        if success_rate >= 0.7:
            final_conclusion = f"✅ DRS仮説は強く支持される (成功率: {100*success_rate:.1f}%)"
        elif success_rate >= 0.5:
            final_conclusion = f"⚠️ DRS仮説は部分的に支持される (成功率: {100*success_rate:.1f}%)"
        else:
            final_conclusion = f"❌ DRS仮説は支持されない (成功率: {100*success_rate:.1f}%)"

        print(f"\n🎯 最終結論: {final_conclusion}")

        # 設定別分析
        print(f"\n📋 設定別性能分析:")
        for config_name in ['DRS-Aggressive', 'DRS-Balanced', 'DRS-Conservative']:
            config_results = [
                r for r in all_results if r['config'] == config_name]
            config_success = sum(1 for r in config_results if r['tradeoff_rating'] in [
                                 'EXCELLENT', 'GOOD'])
            config_avg_quality = np.mean(
                [r['quality_gain'] for r in config_results])
            config_avg_nfe = np.mean([r['nfe_ratio'] for r in config_results])

            print(f"  {config_name}: 成功率 {config_success}/{len(config_results)}, 品質向上 {config_avg_quality:+.3f}, NFE比率 {config_avg_nfe:.3f}x")

        return all_results

    except Exception as e:
        logger.error(f"検証エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = enhanced_drs_validation()
    if results:
        print(f"\n✅ 改善版DRS仮説検証完了")
    else:
        print(f"\n❌ 検証失敗")
