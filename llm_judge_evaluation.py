#!/usr/bin/env python3
"""
LLM-as-a-Judge 評価スクリプト
Gemini 2.0 Flash を用いたペアワイズ評価
"""

import os
import json
import random
import time
from typing import List, Dict, Tuple, Optional
import torch
from tqdm import tqdm

# Google Gemini APIを使用（環境変数にAPIキーが設定されている前提）
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️ google-generativeai パッケージがインストールされていません")
    print("pip install google-generativeai でインストールしてください")
    GEMINI_AVAILABLE = False

from integrated_generation import generate_fast_long, load_model_with_scaling


class LLMJudge:
    """LLM-as-a-Judge 評価クラス"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash"):
        """
        Args:
            api_key: Google API キー（環境変数 GOOGLE_API_KEY からも取得可能）
            model_name: 使用するGeminiモデル名
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai パッケージが必要です")

        # APIキー設定
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY 環境変数またはapi_keyパラメータでAPIキーを設定してください")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

        print(f"✅ LLM Judge 初期化完了: {model_name}")

    def judge_pair(self, prompt: str, output_a: str, output_b: str,
                   criteria: str = "一般的な品質") -> Tuple[str, str]:
        """
        2つの出力をペアワイズ評価

        Args:
            prompt: 元のプロンプト
            output_a: 出力A
            output_b: 出力B
            criteria: 評価基準

        Returns:
            (preferred_output, reasoning): 優れた出力（'A' or 'B'）と理由
        """
        # ランダムに順序を入れ替えてバイアスを除去
        if random.choice([True, False]):
            first, second = output_a, output_b
            labels = ["A", "B"]
        else:
            first, second = output_b, output_a
            labels = ["B", "A"]

        judge_prompt = f"""以下の2つのAI出力を比較し、どちらが優れているかを判断してください。

評価基準: {criteria}

【元のプロンプト】
{prompt}

【出力1】
{first}

【出力2】  
{second}

【指示】
上記の2つの出力を比較し、以下の観点で評価してください：
- 正確性と事実性
- 明確性と理解しやすさ
- 質問への適切な回答
- 有用性

まず簡潔な理由を述べ、最後に「判定: 1」または「判定: 2」で結論してください。"""

        try:
            response = self.model.generate_content(
                judge_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=500
                )
            )

            response_text = response.text.strip()

            # 判定結果を抽出
            if "判定: 1" in response_text:
                preferred = labels[0]
            elif "判定: 2" in response_text:
                preferred = labels[1]
            else:
                # フォールバック：より詳細な解析
                if "1の方が" in response_text or "出力1" in response_text:
                    preferred = labels[0]
                elif "2の方が" in response_text or "出力2" in response_text:
                    preferred = labels[1]
                else:
                    preferred = random.choice(labels)  # ランダム選択

            return preferred, response_text

        except Exception as e:
            print(f"⚠️ 判定エラー: {e}")
            return random.choice(labels), f"エラーのためランダム選択: {e}"

    def evaluate_multiple(self, prompts: List[str], outputs_a: List[str],
                          outputs_b: List[str], criteria: str = "一般的な品質",
                          labels: Tuple[str, str] = ("モデルA", "モデルB")) -> Dict:
        """
        複数のプロンプトに対してペアワイズ評価

        Args:
            prompts: プロンプトリスト
            outputs_a: モデルAの出力リスト
            outputs_b: モデルBの出力リスト
            criteria: 評価基準
            labels: モデルのラベル

        Returns:
            評価結果の辞書
        """
        if not (len(prompts) == len(outputs_a) == len(outputs_b)):
            raise ValueError("プロンプトと出力の数が一致しません")

        results = []
        wins_a = 0

        print(f"🧑‍⚖️ LLM Judge 評価開始: {len(prompts)} 件")

        for i, (prompt, out_a, out_b) in enumerate(tqdm(zip(prompts, outputs_a, outputs_b))):
            preferred, reasoning = self.judge_pair(
                prompt, out_a, out_b, criteria)

            if preferred == "A":
                wins_a += 1
                winner = labels[0]
            else:
                winner = labels[1]

            results.append({
                "prompt": prompt,
                "output_a": out_a,
                "output_b": out_b,
                "preferred": preferred,
                "winner": winner,
                "reasoning": reasoning
            })

            # レート制限対策
            time.sleep(0.5)

        win_rate_a = wins_a / len(prompts)

        evaluation_result = {
            "model_a_label": labels[0],
            "model_b_label": labels[1],
            "total_comparisons": len(prompts),
            "model_a_wins": wins_a,
            "model_b_wins": len(prompts) - wins_a,
            "model_a_win_rate": win_rate_a,
            "model_b_win_rate": 1 - win_rate_a,
            "criteria": criteria,
            "judge_model": self.model_name,
            "detailed_results": results
        }

        return evaluation_result


def generate_test_cases():
    """テスト用のプロンプトを生成"""
    test_prompts = [
        "次の数学問題を解いてください：太郎は毎時12kmで4時間走り、その後毎時6kmで走りました。合計8時間でどれだけの距離を走れますか？",
        "Pythonでリストから重複を除去する方法を3つ教えてください。",
        "日本の四季について短い詩を書いてください。",
        "機械学習の教師あり学習と教師なし学習の違いを説明してください。",
        "健康的な生活習慣について5つのアドバイスをください。",
        "気候変動問題の解決策を3つ提案してください。",
        "小説の主人公の心理描写を含む短い文章を書いてください。",
        "データサイエンスプロジェクトの進め方を段階的に説明してください。"
    ]

    return test_prompts


def compare_configurations():
    """異なる設定での生成結果を比較"""
    print("🆚 設定比較実験")

    # モデル読み込み
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'
    model, tokenizer, config = load_model_with_scaling(
        model_path, scaling_factor=1)

    # テストプロンプト
    test_prompts = generate_test_cases()

    # 設定A: 高速設定
    config_a = {
        "name": "高速設定",
        "steps": 64,
        "block_length": 32,
        "remasking": "random"
    }

    # 設定B: 品質重視設定
    config_b = {
        "name": "品質重視設定",
        "steps": 256,
        "block_length": 32,
        "remasking": "low_confidence"
    }

    outputs_a = []
    outputs_b = []

    print(f"🔄 {len(test_prompts)} プロンプトで生成中...")

    for prompt in tqdm(test_prompts):
        # フォーマット
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(
            formatted_prompt, return_tensors='pt').input_ids.to(model.device)

        # 設定Aで生成
        outputs_a_raw, _, _ = generate_fast_long(
            model=model, prompt=input_ids, gen_length=128, dual_cache=True, **config_a
        )
        result_a = tokenizer.decode(
            outputs_a_raw[0, input_ids.shape[1]:], skip_special_tokens=True)
        outputs_a.append(result_a)

        # 設定Bで生成
        outputs_b_raw, _, _ = generate_fast_long(
            model=model, prompt=input_ids, gen_length=128, dual_cache=True, **config_b
        )
        result_b = tokenizer.decode(
            outputs_b_raw[0, input_ids.shape[1]:], skip_special_tokens=True)
        outputs_b.append(result_b)

    return test_prompts, outputs_a, outputs_b, config_a["name"], config_b["name"]


def run_judge_evaluation():
    """LLM Judge 評価の実行"""
    print("🧑‍⚖️ LLM-as-a-Judge 評価実験")

    # API キー確認
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY 環境変数が設定されていません")
        print("Google AI Studio (https://aistudio.google.com/) でAPIキーを取得し、")
        print("GOOGLE_API_KEY 環境変数に設定してください")
        return None

    try:
        # Judge 初期化
        judge = LLMJudge()

        # テストケース生成
        prompts, outputs_a, outputs_b, label_a, label_b = compare_configurations()

        # 評価実行
        results = judge.evaluate_multiple(
            prompts=prompts,
            outputs_a=outputs_a,
            outputs_b=outputs_b,
            criteria="生成品質と正確性",
            labels=(label_a, label_b)
        )

        # 結果表示
        print("\n📊 LLM Judge 評価結果:")
        print("=" * 50)
        print(f"📝 比較数: {results['total_comparisons']}")
        print(
            f"🏆 {results['model_a_label']} 勝利: {results['model_a_wins']} ({results['model_a_win_rate']:.1%})")
        print(
            f"🏆 {results['model_b_label']} 勝利: {results['model_b_wins']} ({results['model_b_win_rate']:.1%})")

        # 詳細結果の一部表示
        print("\n🔍 詳細結果例:")
        for i, result in enumerate(results['detailed_results'][:3]):
            print(f"\n例 {i+1}:")
            print(f"プロンプト: {result['prompt'][:80]}...")
            print(f"勝者: {result['winner']}")
            print(f"理由: {result['reasoning'][:150]}...")

        # 結果保存
        with open('llm_judge_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n💾 結果が llm_judge_results.json に保存されました")

        return results

    except Exception as e:
        print(f"❌ 評価エラー: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='LLM-as-a-Judge 評価')
    parser.add_argument('--api-key', help='Google API キー')
    parser.add_argument(
        '--model', default='gemini-2.0-flash', help='Judgeモデル')

    args = parser.parse_args()

    if args.api_key:
        os.environ['GOOGLE_API_KEY'] = args.api_key

    run_judge_evaluation()


if __name__ == "__main__":
    main()
