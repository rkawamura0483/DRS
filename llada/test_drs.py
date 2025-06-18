# 改善版DRS検証システム
import logging
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from generate import generate, generate_with_conservative_drs
from model.modeling_llada import LLaDAModelLM
from transformers import AutoTokenizer

# 日本語のコメントはユーザールールに従って保持
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityEvaluator:
    """改善された品質評価システム"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def evaluate_semantic_quality(self, text):
        """意味的品質の評価"""
        if not text or len(text.strip()) == 0:
            return 0.0

        # 1. 基本的な文法構造チェック
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        if len(valid_sentences) == 0:
            return 0.0

        # 2. 反復パターンの検出（改善版）
        words = text.lower().split()
        if len(words) == 0:
            return 0.0

        # 連続する同一語句の検出
        consecutive_repetition_penalty = 0
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                consecutive_repetition_penalty += 1

        # 文レベルでの反復検出
        sentence_repetition_penalty = 0
        sentence_counter = Counter(valid_sentences)
        for count in sentence_counter.values():
            if count > 1:
                sentence_repetition_penalty += count - 1

        # 3. 語彙多様性スコア
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / \
            len(words) if len(words) > 0 else 0

        # 4. 論理的一貫性（キーワード変化の検出）
        consistency_score = 1.0
        # 重要な名詞が変化した場合のペナルティ
        key_nouns = ['garden', 'problem', 'function', 'algorithm']
        found_nouns = [noun for noun in key_nouns if noun in text.lower()]
        if len(found_nouns) > 1:
            # 複数の競合する概念が含まれる場合
            consistency_score *= 0.8

        # 5. 最終スコア計算
        base_score = vocabulary_diversity * consistency_score
        repetition_penalty = (
            consecutive_repetition_penalty + sentence_repetition_penalty) * 0.1

        final_score = max(0, base_score - repetition_penalty)
        return min(1.0, final_score)

    def evaluate_coherence(self, text):
        """文脈の一貫性評価"""
        if not text:
            return 0.0

        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

        if len(valid_sentences) < 2:
            return 0.5  # 短すぎる場合は中立的スコア

        # 文間の語彙的一貫性チェック
        coherence_score = 0.0
        for i in range(len(valid_sentences) - 1):
            words1 = set(valid_sentences[i].lower().split())
            words2 = set(valid_sentences[i+1].lower().split())
            overlap = len(words1 & words2) / max(len(words1), len(words2), 1)
            coherence_score += overlap

        return coherence_score / max(1, len(valid_sentences) - 1)

    def comprehensive_quality_score(self, text):
        """包括的品質スコア"""
        semantic_score = self.evaluate_semantic_quality(text)
        coherence_score = self.evaluate_coherence(text)

        # 重み付き平均
        final_score = 0.7 * semantic_score + 0.3 * coherence_score
        return final_score


def improved_ambiguity_calculation(confidence_scores, threshold, baseline_confidence=None):
    """改善された曖昧度計算"""
    valid_scores = confidence_scores[confidence_scores != -np.inf]
    if len(valid_scores) == 0:
        return 0.0

    # 1. 相対的曖昧度計算（ベースラインとの比較）
    if baseline_confidence is not None:
        baseline_valid = baseline_confidence[baseline_confidence != -np.inf]
        if len(baseline_valid) > 0:
            relative_confidence_drop = (
                baseline_valid.mean() - valid_scores.mean()).item()
            relative_ambiguity = max(0, relative_confidence_drop)
        else:
            relative_ambiguity = 0
    else:
        relative_ambiguity = 0

    # 2. 分散ベースの曖昧度
    variance_ambiguity = valid_scores.var().item()

    # 3. 閾値ベースの曖昧度
    threshold_ambiguity = (valid_scores < threshold).float().mean().item()

    # 統合曖昧度スコア
    combined_ambiguity = (
        0.4 * threshold_ambiguity +
        0.3 * variance_ambiguity +
        0.3 * relative_ambiguity
    )

    return combined_ambiguity


def test_improved_drs_validation():
    """改善版DRS仮説検証"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔬 改善版DRS検証開始 (デバイス: {device})")

    # より適切なテストプロンプト
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

        # 品質評価器の初期化
        evaluator = QualityEvaluator(tokenizer)

        # 改善されたテスト設定
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

            # より適切なテスト条件
            test_conditions = [
                {'t_base': 4, 'threshold': 0.7, 'name': '適度な条件'},
            ]

            for condition in test_conditions:
                print(f"\n{'-'*60}")
                print(
                    f"🧪 {condition['name']} (t_base={condition['t_base']}, threshold={condition['threshold']})")
                print(f"{'-'*60}")

                # ベースライン生成
                print("🎯 ベースライン生成中...")
                baseline_out, baseline_nfe = generate(
                    model, input_ids, steps=total_steps, gen_length=gen_length,
                    block_length=block_length, temperature=0., remasking='low_confidence',
                    mask_id=mask_id
                )

                # 保守的DRS生成
                print("⚡ 保守的DRS生成中...")
                drs_out, drs_nfe, ambiguity_scores = generate_with_conservative_drs(
                    model, input_ids, steps=total_steps, gen_length=gen_length,
                    block_length=block_length, temperature=0.,
                    threshold=condition['threshold'], t_base=condition['t_base'],
                    mask_id=mask_id
                )

                # テキスト抽出
                baseline_text = tokenizer.batch_decode(
                    baseline_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                drs_text = tokenizer.batch_decode(
                    drs_out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

                # 改善された品質評価
                baseline_quality = evaluator.comprehensive_quality_score(
                    baseline_text)
                drs_quality = evaluator.comprehensive_quality_score(drs_text)
                quality_retention = (
                    drs_quality / baseline_quality) * 100 if baseline_quality > 0 else 0

                # NFE効率
                nfe_reduction = ((baseline_nfe - drs_nfe) / baseline_nfe) * 100

                # 曖昧度分析
                max_ambiguity = max(
                    ambiguity_scores) if ambiguity_scores else 0
                ambiguity_variance = np.var(
                    ambiguity_scores) if ambiguity_scores else 0
                meaningful_blocks = sum(
                    1 for score in ambiguity_scores if score > 0.1)

                # 結果出力
                print(f"\n📊 詳細結果:")
                print(f"  🎯 ベースライン:")
                print(
                    f"     NFE: {baseline_nfe}, 品質スコア: {baseline_quality:.3f}")
                print(f"     テキスト: {baseline_text[:100]}...")
                print(f"  ⚡ 保守的DRS:")
                print(f"     NFE: {drs_nfe}, 品質スコア: {drs_quality:.3f}")
                print(f"     テキスト: {drs_text[:100]}...")

                print(f"\n📈 パフォーマンス分析:")
                print(f"  🔄 NFE削減: {nfe_reduction:.1f}%")
                print(f"  💎 品質保持: {quality_retention:.1f}%")
                print(f"  🎭 最大曖昧度: {max_ambiguity:.3f}")
                print(f"  📊 曖昧度分散: {ambiguity_variance:.3f}")
                print(
                    f"  🔍 意味あるブロック: {meaningful_blocks}/{len(ambiguity_scores)}")

                # DRS価値の評価（改善版）
                if (nfe_reduction > 20 and quality_retention > 70 and
                        meaningful_blocks >= 2 and ambiguity_variance > 0.01):
                    drs_value = "✅ TRUE - 有効な動的配分"
                elif (nfe_reduction > 15 and quality_retention > 50):
                    drs_value = "⚠️ PARTIAL - 限定的効果"
                else:
                    drs_value = "❌ FALSE - 効果不明または品質劣化"

                print(f"  🎯 DRS価値評価: {drs_value}")

                # 結果保存
                results.append({
                    'prompt_id': i+1,
                    'condition': condition['name'],
                    'nfe_reduction': nfe_reduction,
                    'quality_retention': quality_retention,
                    'baseline_quality': baseline_quality,
                    'drs_quality': drs_quality,
                    'max_ambiguity': max_ambiguity,
                    'ambiguity_variance': ambiguity_variance,
                    'meaningful_blocks': meaningful_blocks,
                    'drs_value': drs_value
                })

        # 全体的な結論
        print(f"\n{'='*80}")
        print("🎯 改善版検証結果サマリー")
        print(f"{'='*80}")

        successful_cases = sum(1 for r in results if 'TRUE' in r['drs_value'])
        partial_cases = sum(1 for r in results if 'PARTIAL' in r['drs_value'])
        total_cases = len(results)

        avg_nfe_reduction = np.mean([r['nfe_reduction'] for r in results])
        avg_quality_retention = np.mean(
            [r['quality_retention'] for r in results])

        print(f"📊 統計:")
        print(f"  ✅ 完全成功: {successful_cases}/{total_cases}")
        print(f"  ⚠️ 部分成功: {partial_cases}/{total_cases}")
        print(f"  📉 平均NFE削減: {avg_nfe_reduction:.1f}%")
        print(f"  💎 平均品質保持: {avg_quality_retention:.1f}%")

        if successful_cases >= total_cases * 0.5:
            final_conclusion = "✅ DRS仮説は検証された - 適切な条件下で有効"
        elif (successful_cases + partial_cases) >= total_cases * 0.6:
            final_conclusion = "⚠️ DRS仮説は部分的に検証 - さらなる調整が必要"
        else:
            final_conclusion = "❌ DRS仮説は検証されず - 根本的見直しが必要"

        print(f"\n🎯 最終結論: {final_conclusion}")

        return results

    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = test_improved_drs_validation()
    if results:
        print(f"\n✅ 改善版検証完了")
    else:
        print(f"\n❌ 検証失敗")
