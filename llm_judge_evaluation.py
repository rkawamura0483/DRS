#!/usr/bin/env python3
"""
LLM-as-a-Judge 評価スクリプト
Fast-dLLM × LongLLaDA 統合実験用
Gemini 2.0 Flash を用いたペアワイズ評価
"""

import os
import json
import random
import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
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

from integrated_generation import generate_fast_long, load_model_with_scaling, generate_no_cache


class LLMJudge:
    """LLM-as-a-Judge 評価クラス（Fast-dLLM × LongLLaDA 統合実験用）"""

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
        READMEで示されている評価フローに従った2つの出力のペアワイズ評価
        - シャッフルして非公開IDを付与
        - 温度0で一貫性を保つ
        """
        # ランダムに順序を入れ替えてバイアスを除去（READMEの評価フロー）
        if random.choice([True, False]):
            first, second = output_a, output_b
            labels = ["A", "B"]
        else:
            first, second = output_b, output_a
            labels = ["B", "A"]

        judge_prompt = f"""## プロンプト
{prompt}

### 回答A
{first}

### 回答B
{second}

# 指示
評価基準「{criteria}」に基づいて、以下の観点で2つの回答を比較してください：

**評価観点:**
- 正確性と事実性
- 明確性と理解しやすさ  
- 質問への適切な回答
- 有用性
- 長文処理能力（該当する場合）
- 一貫性

まず簡潔な理由を述べ、最後に優れている方のラベル ('A' or 'B') のみを出力してください。"""

        try:
            response = self.model.generate_content(
                judge_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,  # READMEで推奨されている温度0
                    max_output_tokens=500
                )
            )

            response_text = response.text.strip()

            # 判定結果を抽出（READMEのフローに従う）
            if response_text.endswith('A'):
                preferred = labels[0]
            elif response_text.endswith('B'):
                preferred = labels[1]
            elif "A" in response_text.split()[-3:]:  # 最後の方でAが言及
                preferred = labels[0]
            elif "B" in response_text.split()[-3:]:  # 最後の方でBが言及
                preferred = labels[1]
            else:
                # フォールバック
                preferred = random.choice(labels)

            return preferred, response_text

        except Exception as e:
            print(f"⚠️ 判定エラー: {e}")
            return random.choice(labels), f"エラーのためランダム選択: {e}"

    def calculate_j_score(self, prompts: List[str], outputs_a: List[str],
                          outputs_b: List[str], criteria: str, num_repeats: int = 3) -> float:
        """
        Judge Consistency (J-score) を計算
        同一プロンプトセットを複数回評価し、一貫性を測定
        """
        print(f"🔄 Judge Consistency 計算中 (評価回数: {num_repeats})")

        all_preferences = []

        for repeat in range(num_repeats):
            preferences = []
            for prompt, out_a, out_b in tqdm(zip(prompts, outputs_a, outputs_b),
                                             desc=f"評価 {repeat+1}/{num_repeats}"):
                preferred, _ = self.judge_pair(prompt, out_a, out_b, criteria)
                preferences.append(1 if preferred == "A" else 0)
                time.sleep(0.3)  # レート制限対策

            all_preferences.append(preferences)

        # 一貫性スコア計算（各プロンプトでの評価の分散の平均）
        consistency_scores = []
        for i in range(len(prompts)):
            votes = [prefs[i] for prefs in all_preferences]
            # 分散が小さいほど一貫性が高い
            variance = np.var(votes)
            consistency_scores.append(1 - variance)  # 1に近いほど一貫

        j_score = np.mean(consistency_scores)
        print(f"📊 Judge Consistency (J-score): {j_score:.3f}")

        return j_score

    def evaluate_multiple(self, prompts: List[str], outputs_a: List[str],
                          outputs_b: List[str], criteria: str = "一般的な品質",
                          labels: Tuple[str, str] = ("モデルA", "モデルB"),
                          calculate_consistency: bool = True) -> Dict:
        """
        複数のプロンプトに対してペアワイズ評価（READMEの評価フロー準拠）
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

        # Judge Consistency 計算
        if calculate_consistency and len(prompts) >= 3:
            try:
                j_score = self.calculate_j_score(prompts[:min(5, len(prompts))],
                                                 outputs_a[:min(
                                                     5, len(outputs_a))],
                                                 outputs_b[:min(5, len(outputs_b))], criteria)
                evaluation_result["j_score"] = j_score
            except Exception as e:
                print(f"⚠️ J-score 計算エラー: {e}")
                evaluation_result["j_score"] = None

        return evaluation_result


def generate_test_cases():
    """READMEの研究目的に沿ったテスト用プロンプトを生成"""
    # 基本的なプロンプト
    basic_prompts = [
        "次の数学問題を解いてください：太郎は毎時12kmで4時間走り、その後毎時6kmで走りました。合計8時間でどれだけの距離を走れますか？",
        "Pythonでリストから重複を除去する方法を3つ教えてください。",
        "日本の四季について短い詩を書いてください。",
        "機械学習の教師あり学習と教師なし学習の違いを説明してください。",
        "健康的な生活習慣について5つのアドバイスをください。",
    ]

    # 長文処理テスト用プロンプト（メモリ制限を考慮して短縮）
    long_context_prompts = [
        "次の長い物語の要約を書いてください：" + "昔々、ある村に勇敢な少年がいました。" * 50 + " この物語の主要なテーマは何ですか？",
        "以下の技術文書の要点を整理してください：" + "機械学習は現代のAI技術の基盤です。" * 75 + " 最も重要なポイントを3つ挙げてください。",
        "この長いコードの動作を説明してください：\n```python\n" + "# 重要な処理\nresult = process_data()\n" *
        40 + "```\nこのコードの目的は何ですか？",
    ]

    return {
        "basic": basic_prompts,
        "long_context": long_context_prompts,
        "all": basic_prompts + long_context_prompts
    }


def niah_test(model, tokenizer, needle_info: str = "重要な情報：答えは42です。",
              question: str = "重要な情報は何ですか？", context_length: int = 4000) -> Dict:
    """
    NIAH (Needle in a Haystack) テスト実装（READMEで言及されているテスト）

    Args:
        model: 評価対象モデル
        tokenizer: トークナイザ
        needle_info: 探すべき情報
        question: 質問
        context_length: コンテキスト長（トークン数の目安）

    Returns:
        NAIHテスト結果
    """
    print(f"🔍 NIAH テスト開始 (目標コンテキスト長: {context_length})")

    # メモリ節約のため最大コンテキスト長を制限
    max_safe_length = min(context_length, 8000)  # 8kトークンまでに制限

    # ダミーコンテキスト生成（より効率的な方法）
    dummy_text = "これは重要ではない情報です。自然言語処理技術の発展により、大規模言語モデルが注目されています。"
    dummy_tokens = tokenizer(
        dummy_text, return_tensors='pt').input_ids.shape[1]

    # 必要な繰り返し回数を計算（トークン数ベース）
    repeat_count = max(1, max_safe_length // dummy_tokens // 2)  # 安全マージン
    haystack = (dummy_text + " ") * repeat_count

    # needleを途中に埋め込み
    haystack_words = haystack.split()
    insert_position = len(haystack_words) // 2
    haystack_words.insert(insert_position, needle_info)

    full_context = " ".join(haystack_words) + f"\n\n質問：{question}\n回答："

    # トークナイズして長さを確認
    input_ids = tokenizer(
        full_context, return_tensors='pt', truncation=True, max_length=max_safe_length
    ).input_ids.to(model.device)
    actual_length = input_ids.shape[1]

    print(f"📏 実際のコンテキスト長: {actual_length} トークン (制限: {max_safe_length})")

    # 生成実行
    try:
        outputs, nfe, metrics = generate_fast_long(
            model=model,
            prompt=input_ids,
            gen_length=64,  # 生成長を短くしてメモリ節約
            steps=32,       # ステップ数も削減
            block_length=16,  # 小さなブロック
            temperature=0.0,
            remasking='low_confidence',
            dual_cache=False  # デュアルキャッシュを無効化してメモリ節約
        )

        result = tokenizer.decode(
            outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

        # 成功判定
        success = "42" in result

        niah_result = {
            "success": success,
            "context_length": actual_length,
            "needle_position": insert_position / len(haystack_words),
            "generated_answer": result.strip(),
            "target_answer": "42",
            "needle_info": needle_info,
            "question": question,
            "nfe": nfe,
            "speed": metrics.get('tokens_per_second', 0)
        }

        print(f"✅ NIAH 成功: {success}")
        print(f"🎯 生成された回答: {result.strip()[:100]}...")

        return niah_result

    except Exception as e:
        print(f"❌ NIAH テストエラー: {e}")
        return {
            "success": False,
            "error": str(e),
            "context_length": actual_length,
            "needle_info": needle_info,
            "question": question
        }


def compare_scaling_factors():
    """
    READMEのスケール係数表に基づく比較実験
    8k, 16k, 24k, 32k の各設定での性能評価
    """
    print("🔢 RoPE スケーリング係数比較実験")

    # モデル読み込み
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'

    # READMEのスケール係数表
    scaling_configs = [
        {"name": "8k設定", "target_length": 8000, "scaling_factor": 4},
        {"name": "16k設定", "target_length": 16000, "scaling_factor": 14},
        {"name": "24k設定", "target_length": 24000, "scaling_factor": 31},
        {"name": "32k設定", "target_length": 32000, "scaling_factor": 55},
    ]

    # テストプロンプト
    test_prompts = generate_test_cases()["long_context"][:3]  # 長文テスト用

    results = {}

    for config in scaling_configs:
        print(f"\n🧪 {config['name']} 評価中...")

        try:
            # スケーリング適用モデルの読み込み
            model, tokenizer, model_config = load_model_with_scaling(
                model_path, scaling_factor=config["scaling_factor"])

            outputs = []
            niah_results = []

            for prompt in test_prompts:
                # フォーマット
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                input_ids = tokenizer(
                    formatted_prompt, return_tensors='pt').input_ids.to(model.device)

                # 生成実行（品質改善のためパラメータ調整）
                output_ids, nfe, metrics = generate_fast_long(
                    model=model,
                    prompt=input_ids,
                    gen_length=256,
                    steps=128,  # ステップ数を増加して品質向上
                    block_length=16,  # ブロック長を小さくして精度向上
                    temperature=0.3,  # 適度な温度で自然な生成
                    remasking='low_confidence',
                    dual_cache=True
                )

                result = tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
                outputs.append(result)

            # NAIHテスト（メモリ制限で実用的な長さに調整）
            safe_context_length = min(
                config["target_length"]//4, 4000)  # より安全な長さ
            niah_result = niah_test(
                model, tokenizer, context_length=safe_context_length)
            niah_results.append(niah_result)

            results[config["name"]] = {
                "outputs": outputs,
                "niah_results": niah_results,
                "scaling_factor": config["scaling_factor"],
                "target_length": config["target_length"]
            }

            # 強制的なメモリクリア
            del model, outputs, niah_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import gc
            gc.collect()

            print(f"✅ {config['name']} 完了 - メモリクリア実行")

        except Exception as e:
            print(f"❌ {config['name']} エラー: {e}")
            results[config["name"]] = {"error": str(e)}

    return test_prompts, results


def compare_vanilla_vs_fast_dllm():
    """
    単純なLongLLaDA vs LongLLaDA + Fast-dLLM の比較
    研究の核心：Fast-dLLMの拡散生成機構の効果を検証
    """
    print("🆚 LongLLaDA vs LongLLaDA + Fast-dLLM 比較実験")

    # モデル読み込み
    model_path = 'GSAI-ML/LLaDA-8B-Instruct'
    model, tokenizer, config = load_model_with_scaling(
        model_path, scaling_factor=1)

    # 互換性パッチ: LLaDAModelLM が forward の未知キーワードを拒否するため generate でエラーになる
    def _patch_forward_for_generate(m):
        if getattr(m, "_patched_for_generate", False):
            return  # 既にパッチ済み
        original_forward = m.forward

        def wrapped_forward(*args, **kwargs):
            # transformers>=4.40 で追加された cache_position などを除去
            kwargs.pop('cache_position', None)
            return original_forward(*args, **kwargs)
        m.forward = wrapped_forward
        m._patched_for_generate = True
    _patch_forward_for_generate(model)

    # テストプロンプト
    test_prompts = generate_test_cases()["basic"]

    outputs_vanilla = []  # 標準LongLLaDA
    outputs_fast_dllm = []  # LongLLaDA + Fast-dLLM
    metrics_vanilla = []
    metrics_fast_dllm = []

    print(f"🔄 {len(test_prompts)} プロンプトで両方式生成中...")

    for prompt in tqdm(test_prompts):
        # フォーマット
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(
            formatted_prompt, return_tensors='pt').input_ids.to(model.device)

        # 1. 標準的なLongLLaDA生成（LongLLaDAの独自拡散生成）
        start_time = time.time()
        try:
            # integrated_generation.py内のgenerate_no_cache関数を使用（標準的な拡散生成）
            from integrated_generation import generate_no_cache

            # 標準拡散生成パラメータ（キャッシュなし）
            vanilla_outputs = generate_no_cache(
                model=model,
                prompt=input_ids,
                steps=32,  # 基本的なステップ数
                gen_length=128,  # 生成長
                block_length=128,  # 大きなブロック（標準的な設定）
                temperature=0.0,  # 決定的生成
                remasking='low_confidence',  # 信頼度ベースリマスキング
                mask_id=126336,  # LongLLaDAのマスクトークンID
                scaling_factor=1
            )
            vanilla_time = time.time() - start_time

            # 生成部分のみを抽出し、適切にデコード
            result_vanilla = tokenizer.decode(
                vanilla_outputs[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        except Exception as e:
            print(f"⚠️ 標準LongLLaDA生成エラー: {e}")
            print("🔄 フォールバック: より基本的なパラメータで再試行...")
            try:
                # より基本的なパラメータで再試行
                vanilla_outputs = generate_no_cache(
                    model=model,
                    prompt=input_ids,
                    steps=16,  # ステップ数削減
                    gen_length=64,  # 生成長削減
                    block_length=64,  # 小さなブロック
                    temperature=0.1,  # 若干のランダム性
                    remasking='random',  # ランダムマスキングに変更
                    mask_id=126336,
                    scaling_factor=1
                )
                vanilla_time = time.time() - start_time
                result_vanilla = tokenizer.decode(
                    vanilla_outputs[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                print("✅ フォールバック生成成功")
            except Exception as e2:
                print(f"❌ フォールバック生成も失敗: {e2}")
                result_vanilla = f"拡散生成エラー: {str(e2)}"
                vanilla_time = time.time() - start_time
                vanilla_tokens = 0  # エラー時はトークン数0

        outputs_vanilla.append(result_vanilla)

        # 標準生成のメトリクス
        if 'vanilla_outputs' in locals() and vanilla_outputs is not None:
            vanilla_tokens = vanilla_outputs.shape[1] - input_ids.shape[1]
        else:
            vanilla_tokens = 0  # エラー時はトークン数0

        metrics_vanilla.append({
            "generation_time": vanilla_time,
            "tokens_generated": vanilla_tokens,
            "tokens_per_second": vanilla_tokens / vanilla_time if vanilla_time > 0 else 0,
            "method": "llada_diffusion_generation"
        })

        # 2. LongLLaDA + Fast-dLLM（拡散生成機構）
        try:
            outputs_fast_raw, nfe, metrics_fast = generate_fast_long(
                model=model,
                prompt=input_ids,
                gen_length=128,
                steps=64,  # ステップ数を削減してメモリ節約
                block_length=32,
                temperature=0.0,
                remasking='low_confidence',
                dual_cache=True
            )

            result_fast = tokenizer.decode(
                outputs_fast_raw[0, input_ids.shape[1]                                 :], skip_special_tokens=True
            ).strip()

            # Fast-dLLMのメトリクス
            metrics_fast["nfe"] = nfe
            metrics_fast["method"] = "fast_dllm_diffusion"

        except Exception as e:
            print(f"⚠️ Fast-dLLM生成エラー: {e}")
            result_fast = f"Fast-dLLM生成エラー: {str(e)}"
            metrics_fast = {
                "generation_time": 0,
                "tokens_generated": 0,
                "tokens_per_second": 0,
                "nfe": 0,
                "method": "fast_dllm_diffusion"
            }

        outputs_fast_dllm.append(result_fast)
        metrics_fast_dllm.append(metrics_fast)

        # 各プロンプトごとにメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (test_prompts, outputs_vanilla, outputs_fast_dllm,
            "標準LongLLaDA", "LongLLaDA+Fast-dLLM",
            metrics_vanilla, metrics_fast_dllm)


def run_comprehensive_evaluation():
    """READMEの研究目的に沿った包括的な評価実験"""
    print("🎯 Fast-dLLM × LongLLaDA 包括的評価実験")

    # API キー確認
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY 環境変数が設定されていません")
        print("Google AI Studio (https://aistudio.google.com/) でAPIキーを取得し、")
        print("GOOGLE_API_KEY 環境変数に設定してください")
        return None

    try:
        # Judge 初期化
        judge = LLMJudge()

        all_results = {}

        # 1. LongLLaDA vs Fast-dLLM 比較
        print("\n📝 LongLLaDA vs Fast-dLLM 比較評価")
        prompts, outputs_a, outputs_b, label_a, label_b, metrics_a, metrics_b = compare_vanilla_vs_fast_dllm()

        # メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"📋 比較対象: {len(outputs_a)} 件のプロンプト")
        print(
            f"📊 A（{label_a}）の空出力数: {sum(1 for x in outputs_a if not x.strip())}")
        print(
            f"📊 B（{label_b}）の空出力数: {sum(1 for x in outputs_b if not x.strip())}")

        basic_eval = judge.evaluate_multiple(
            prompts=prompts,
            outputs_a=outputs_a,
            outputs_b=outputs_b,
            criteria="生成品質と正確性",
            labels=(label_a, label_b),
            calculate_consistency=True
        )

        # 速度情報を追加
        avg_speed_a = np.mean([m.get('tokens_per_second', 0)
                              for m in metrics_a])
        avg_speed_b = np.mean([m.get('tokens_per_second', 0)
                              for m in metrics_b])
        basic_eval["average_speed_a"] = avg_speed_a
        basic_eval["average_speed_b"] = avg_speed_b

        all_results["basic_comparison"] = basic_eval

        # 2. スケーリング係数比較（簡易版）
        print("\n🔢 RoPE スケーリング比較評価")
        try:
            scaling_prompts, scaling_results = compare_scaling_factors()

            # 16k vs 32k の比較例
            if "16k設定" in scaling_results and "32k設定" in scaling_results:
                scaling_eval = judge.evaluate_multiple(
                    prompts=scaling_prompts,
                    outputs_a=scaling_results["16k設定"].get("outputs", []),
                    outputs_b=scaling_results["32k設定"].get("outputs", []),
                    criteria="長文処理能力と品質",
                    labels=("16k設定", "32k設定"),
                    calculate_consistency=False  # 時間短縮のため
                )
                all_results["scaling_comparison"] = scaling_eval

            all_results["scaling_details"] = scaling_results

        except Exception as e:
            print(f"⚠️ スケーリング比較スキップ: {e}")

        # 3. 結果表示
        print("\n📊 包括的評価結果:")
        print("=" * 60)

        if "basic_comparison" in all_results:
            basic = all_results["basic_comparison"]
            print(f"📝 基本比較:")
            print(
                f"  {basic['model_a_label']}: {basic['model_a_win_rate']:.1%} 勝率 ({basic['average_speed_a']:.1f} tok/s)")
            print(
                f"  {basic['model_b_label']}: {basic['model_b_win_rate']:.1%} 勝率 ({basic['average_speed_b']:.1f} tok/s)")
            if basic.get('j_score'):
                print(f"  Judge Consistency: {basic['j_score']:.3f}")

        if "scaling_comparison" in all_results:
            scaling = all_results["scaling_comparison"]
            print(f"\n🔢 スケーリング比較:")
            print(
                f"  {scaling['model_a_label']}: {scaling['model_a_win_rate']:.1%} 勝率")
            print(
                f"  {scaling['model_b_label']}: {scaling['model_b_win_rate']:.1%} 勝率")

        # 4. 結果保存
        output_file = 'comprehensive_evaluation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 結果が {output_file} に保存されました")

        # 最終メモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        import gc
        gc.collect()

        return all_results

    except Exception as e:
        print(f"❌ 評価エラー: {e}")
        import traceback
        traceback.print_exc()

        # エラー時もメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return None


def run_judge_evaluation():
    """従来のLLM Judge 評価（後方互換性のため残存）"""
    return run_comprehensive_evaluation()


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fast-dLLM × LongLLaDA LLM-as-a-Judge 評価')
    parser.add_argument('--api-key', help='Google API キー')
    parser.add_argument('--model', default='gemini-2.0-flash', help='Judgeモデル')
    parser.add_argument('--eval-type', choices=['basic', 'comprehensive', 'niah'],
                        default='comprehensive', help='評価タイプ')

    args = parser.parse_args()

    if args.api_key:
        os.environ['GOOGLE_API_KEY'] = args.api_key

    if args.eval_type == 'comprehensive':
        run_comprehensive_evaluation()
    elif args.eval_type == 'basic':
        run_judge_evaluation()
    elif args.eval_type == 'niah':
        # NAIHテストのみ実行
        print("🔍 NIAH テスト単体実行")
        model_path = 'GSAI-ML/LLaDA-8B-Instruct'
        model, tokenizer, _ = load_model_with_scaling(
            model_path, scaling_factor=14)
        result = niah_test(model, tokenizer)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
