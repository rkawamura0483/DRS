# 改善版DRS検証システム
import logging
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from generate import generate_with_drs, generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer

# 日本語のコメントはユーザールールに従って保持
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedQualityEvaluator:
    """大幅改善された品質評価システム"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def check_semantic_consistency(self, prompt, baseline_text, drs_text):
        """意味的一貫性の評価 - 最重要指標"""
        # プロンプトから重要なキーワードを抽出
        prompt_lower = prompt.lower()
        baseline_lower = baseline_text.lower()
        drs_lower = drs_text.lower()

        # 重要なキーワードの一致度チェック
        if "rectangular" in prompt_lower and "garden" in prompt_lower:
            # 長方形の庭の問題
            baseline_has_rectangle = any(word in baseline_lower for word in [
                                         "rectangular", "rectangle", "length", "width"])
            drs_has_rectangle = any(word in drs_lower for word in [
                                    "rectangular", "rectangle", "length", "width"])

            # 円の概念が混入していないかチェック
            baseline_has_circle = any(word in baseline_lower for word in [
                                      "circle", "radius", "π", "pi"])
            drs_has_circle = any(word in drs_lower for word in [
                                 "circle", "radius", "π", "pi"])

            if baseline_has_rectangle and not baseline_has_circle:
                if drs_has_rectangle and not drs_has_circle:
                    return 1.0  # 完全一致
                elif drs_has_circle:
                    return 0.0  # 重大な意味的エラー
                else:
                    return 0.5  # 部分的

        elif "python" in prompt_lower and "function" in prompt_lower:
            # プログラミング問題
            baseline_has_code = "def " in baseline_lower and "return" in baseline_lower
            drs_has_code = "def " in drs_lower and "return" in drs_lower

            if baseline_has_code:
                if drs_has_code:
                    return 1.0
                else:
                    return 0.0  # コードが生成されていない

        # 一般的な一貫性チェック
        baseline_words = set(baseline_lower.split())
        drs_words = set(drs_lower.split())
        overlap = len(baseline_words & drs_words) / max(len(baseline_words), 1)

        return min(1.0, overlap)

    def evaluate_repetition_penalty(self, text):
        """反復ペナルティの評価"""
        if not text:
            return 1.0

        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(valid_sentences) <= 1:
            return 1.0

        # 完全に同一の文の検出
        sentence_counts = Counter(valid_sentences)
        max_repetition = max(sentence_counts.values())

        if max_repetition >= 3:
            return 0.1  # 重大な反復
        elif max_repetition == 2:
            return 0.5  # 軽度の反復

        # 連続する同一語句の検出
        words = text.split()
        consecutive_count = 0
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                consecutive_count += 1

        if consecutive_count > 5:
            return 0.1
        elif consecutive_count > 2:
            return 0.5

        return 1.0

    def evaluate_completeness(self, text, expected_elements):
        """完全性の評価"""
        if not text:
            return 0.0

        text_lower = text.lower()
        found_elements = sum(
            1 for element in expected_elements if element.lower() in text_lower)

        return found_elements / max(len(expected_elements), 1)

    def comprehensive_quality_score(self, prompt, baseline_text, drs_text):
        """包括的品質スコア - 意味的一貫性を最重要視"""
        # 1. 意味的一貫性 (最重要 - 60%の重み)
        semantic_consistency = self.check_semantic_consistency(
            prompt, baseline_text, drs_text)

        # 2. 反復ペナルティ (30%の重み)
        repetition_score = self.evaluate_repetition_penalty(drs_text)

        # 3. 基本的な完全性 (10%の重み)
        drs_length = len(drs_text.split())
        baseline_length = len(baseline_text.split())
        length_ratio = min(
            1.0, drs_length / max(baseline_length, 1)) if baseline_length > 0 else 0

        # 重み付き最終スコア
        final_score = (
            0.6 * semantic_consistency +
            0.3 * repetition_score +
            0.1 * length_ratio
        )

        return final_score, {
            'semantic_consistency': semantic_consistency,
            'repetition_score': repetition_score,
            'length_ratio': length_ratio
        }


def enhanced_ambiguity_calculation_with_context(confidence_scores, threshold, prompt_context=None):
    """文脈を考慮した改善された曖昧度計算"""
    valid_scores = confidence_scores[confidence_scores != -np.inf]
    if len(valid_scores) == 0:
        return 0.0

    # 基本的な閾値ベース曖昧度
    threshold_ambiguity = (valid_scores < threshold).float().mean().item()

    # 分散ベースの曖昧度（不安定性指標）
    variance_ambiguity = min(1.0, valid_scores.var().item() * 10)  # スケール調整

    # 統合曖昧度スコア（より保守的に）
    combined_ambiguity = 0.7 * threshold_ambiguity + 0.3 * variance_ambiguity

    return combined_ambiguity


def test_drs_validation():
    """DRS仮説検証システム"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔬 DRS仮説検証開始 (デバイス: {device})")

    # 検証用プロンプト
    test_prompts = [
        "Calculate the area of a rectangular garden with length 15 meters and width 8 meters.",
        "Write a Python function to find the factorial of a number using recursion.",
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

        # マスクID取得
        mask_id = tokenizer.mask_token_id or model.config.mask_token_id or 126336
        print(f"✅ モデルロード完了 (mask_id={mask_id})")

        evaluator = ImprovedQualityEvaluator(tokenizer)

        gen_length = 256
        block_length = 32
        total_steps = 128

        results = []

        for i, prompt in enumerate(test_prompts):
            print(f"\n{'='*80}")
            print(f"📝 テストプロンプト {i+1}: {prompt}")
            print(f"{'='*80}")

            m = [{"role": "user", "content": prompt}]
            prompt_formatted = tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt_formatted)['input_ids']
            input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

            # --- ベースライン (fast-dLLM) ---
            print("\n🎯 ベースライン (fast-dLLM: generate_with_dual_cache) 生成中...")
            baseline_out, baseline_nfe = generate_with_dual_cache(
                model, input_ids, steps=total_steps, gen_length=gen_length,
                block_length=block_length, temperature=0., remasking='low_confidence',
                mask_id=mask_id
            )
            baseline_text = tokenizer.batch_decode(
                baseline_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            # --- 提案手法 (DRS) ---
            # DRSのパラメータ設定を変えて実験
            test_conditions = [
                {'t_base': 16, 'threshold': 0.9,
                    'name': 'DRS (t_base=16, 品質重視)'},
                {'t_base': 8, 'threshold': 0.8,
                    'name': 'DRS (t_base=8, バランス)'},
                {'t_base': 4, 'threshold': 0.7,
                    'name': 'DRS (t_base=4, 速度重視)'},
            ]

            for condition in test_conditions:
                print(f"\n{'-'*60}")
                print(f"🧪 {condition['name']}")
                print(f"{'-'*60}")

                print("⚡ DRS (generate_with_drs) 生成中...")
                drs_out, drs_nfe, ambiguity_scores = generate_with_drs(
                    model, input_ids, steps=total_steps, gen_length=gen_length,
                    block_length=block_length, temperature=0.,
                    threshold=condition['threshold'], t_base=condition['t_base'],
                    mask_id=mask_id
                )
                drs_text = tokenizer.batch_decode(
                    drs_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

                # 品質評価
                drs_quality, quality_details = evaluator.comprehensive_quality_score(
                    prompt, baseline_text, drs_text)

                # NFE効率
                # NFEは必ずしも削減されるわけではなく、同程度の予算で品質向上を目指す
                nfe_diff = drs_nfe - baseline_nfe

                # 詳細な結果出力
                print(f"\n📊 詳細結果:")
                print(f"  🎯 ベースライン (fast-dLLM):")
                print(f"     NFE: {baseline_nfe}")
                print(f"     テキスト: {baseline_text[:200]}...")
                print(f"  ⚡ 提案手法 ({condition['name']}):")
                print(f"     NFE: {drs_nfe} (ベースライン比: {nfe_diff:+d})")
                print(f"     テキスト: {drs_text[:200]}...")

                print(f"\n📈 品質分析:")
                print(f"  💎 総合品質スコア: {drs_quality:.3f}")
                print(
                    f"     - 意味的一貫性: {quality_details['semantic_consistency']:.3f}")
                print(
                    f"     - 反復ペナルティ: {quality_details['repetition_score']:.3f}")
                print(f"     - 長さ比率: {quality_details['length_ratio']:.3f}")

                # 仮説検証
                # DRSは、NFEをわずかに増加させることで、品質を大幅に向上させることを目指す
                quality_improved = drs_quality > evaluator.comprehensive_quality_score(
                    prompt, baseline_text, baseline_text)[0] * 1.05  # 5%以上の改善
                nfe_efficient = drs_nfe < total_steps * 1.2  # NFEの増加が20%以内に収まっている

                if quality_improved and nfe_efficient:
                    drs_value = "✅ TRUE - 少ないNFE増加で品質が大幅に向上"
                elif quality_improved:
                    drs_value = "⚠️ PARTIAL - 品質は向上したが、NFEが増えすぎた"
                elif not quality_improved and nfe_efficient:
                    drs_value = "❌ FALSE - 品質が向上せず、DRSの効果が見られない"
                else:
                    drs_value = "❌ FALSE - 品質が向上せず、NFEも増加した"

                print(f"  🎯 DRS仮説評価: {drs_value}")
                print(f"  - 曖昧度スコア: {[f'{s:.3f}' for s in ambiguity_scores]}")

                # 結果保存
                results.append({
                    'prompt_id': i+1,
                    'condition': condition['name'],
                    'nfe_diff': nfe_diff,
                    'total_quality': drs_quality,
                    'semantic_consistency': quality_details['semantic_consistency'],
                    'repetition_score': quality_details['repetition_score'],
                    'length_ratio': quality_details['length_ratio'],
                    'drs_value': drs_value
                })

        # 全体的な結論
        print(f"\n{'='*80}")
        print("🎯 DRS仮説検証結果サマリー")
        print(f"{'='*80}")

        successful_cases = sum(1 for r in results if 'TRUE' in r['drs_value'])
        partial_cases = sum(1 for r in results if 'PARTIAL' in r['drs_value'])
        total_cases = len(results)

        avg_nfe_diff = np.mean([r['nfe_diff'] for r in results])
        avg_total_quality = np.mean([r['total_quality'] for r in results])
        avg_semantic_consistency = np.mean(
            [r['semantic_consistency'] for r in results])

        print(f"📊 統計:")
        print(f"  ✅ 完全成功: {successful_cases}/{total_cases}")
        print(f"  ⚠️ 部分成功: {partial_cases}/{total_cases}")
        print(f"  📉 平均NFE差: {avg_nfe_diff:.1f}")
        print(f"  💎 平均総合品質: {avg_total_quality:.3f}")
        print(f"  🎭 平均意味的一貫性: {avg_semantic_consistency:.3f}")

        # 研究価値の最終判定（より厳格な基準）
        semantic_success_rate = sum(
            1 for r in results if r['semantic_consistency'] >= 0.8) / total_cases

        if successful_cases >= total_cases * 0.5 and semantic_success_rate >= 0.8:
            final_conclusion = "✅ DRS仮説は検証された - 高品質維持&効率向上実現"
        elif semantic_success_rate >= 0.6:
            final_conclusion = "⚠️ DRS仮説は部分的に検証 - 意味保持は良好だが効率要改善"
        else:
            final_conclusion = "❌ DRS仮説は検証されず - 意味的一貫性に重大な問題"

        print(f"\n🎯 最終結論: {final_conclusion}")

        # 改善提案
        if semantic_success_rate < 0.8:
            print(f"\n💡 改善提案:")
            print(f"  1. 再マスク閾値をさらに保守的に設定 (0.95+)")
            print(f"  2. 完成ブロックの再マスクを完全に禁止")
            print(f"  3. ブロック間の文脈継続性メカニズムを追加")
            print(f"  4. 意味的検証ステップを生成プロセスに組み込み")

        return results

    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = test_drs_validation()
    if results:
        print(f"\n✅ DRS仮説検証完了")
    else:
        print(f"\n❌ 検証失敗")
