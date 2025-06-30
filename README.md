# Fast-dLLM × LongLLaDA 比較検証実験（Google Colab 対応）

## 🚀 超簡単 Colab セットアップ（推奨）

**3つのコマンドだけで開始！**

```python
# 1. リポジトリをクローン
!git clone https://github.com/your-username/your-repo-name.git
%cd your-repo-name

# 2. 自動セットアップ（Fast-dLLM と LongLLaDA が自動クローンされます）
!python setup_colab.py

# 3. デモ実行
!python quick_start.py
```

> **✨ 新機能**: `integrated_generation.py` が **自動的に** Fast-dLLM と LongLLaDA をクローンします！手動セットアップは不要です。

---

Fast-dLLM の拡散生成機構と LongLLaDA の長文拡張機構を統合し、**標準的なLongLLaDA** と **LongLLaDA + Fast-dLLM** の性能を比較検証するための実験環境です。Google Colab での実行に最適化されています。

## 📋 研究目的

本プロジェクトの核心は以下の比較検証です：

| 手法 | 生成方式 | 期待される特徴 |
|------|----------|----------------|
| **標準LongLLaDA** | Hugging Face標準生成 | 高品質だが低速 |
| **LongLLaDA + Fast-dLLM** | 拡散生成機構 | 高速化と品質維持の両立 |

### 🎯 検証項目
- **生成品質**: LLM-as-a-Judge による客観評価
- **生成速度**: トークン/秒の定量比較  
- **長文対応**: RoPEスケーリングでの長文処理能力
- **Judge一貫性**: 評価の安定性検証

> **🔬 研究仮説**: Fast-dLLMの拡散生成機構により、品質を保ちながら生成速度が向上する

---

## 🚀 Google Colab セットアップ

### 1. 環境準備
```python
# Colab での GPU 確認
!nvidia-smi

# 必要パッケージのインストール
!pip install transformers torch accelerate flash-attn==2.3.3
!pip install huggingface_hub einops

# リポジトリのクローン
!git clone https://github.com/your-repo/Fast-dLLM.git
import sys
sys.path.append('/content/Fast-dLLM')
```

### 2. モデルダウンロード
```python
from huggingface_hub import snapshot_download
import torch

# LLaDA-8B-Instruct のダウンロード
model_path = snapshot_download(
    repo_id='GSAI-ML/LLaDA-8B-Instruct',
    local_dir='/content/llada-8b',
    revision='main'
)
print(f"モデルパス: {model_path}")
```

---

## 💡 比較実験の基本的な使い方

### 🆚 標準LongLLaDA vs LongLLaDA + Fast-dLLM 比較
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from integrated_generation import generate_fast_long, load_model_with_scaling

# モデル・トークナイザの読み込み
model, tokenizer, config = load_model_with_scaling(
    'GSAI-ML/LLaDA-8B-Instruct', 
    scaling_factor=1
)

# テストプロンプト
prompt = "次の数学問題を解いてください：太郎は毎時12kmで4時間走り、その後毎時6kmで走りました。合計8時間でどれだけの距離を走れますか？"

# チャットテンプレートの適用
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
input_ids = tokenizer(formatted_prompt, return_tensors='pt').input_ids.to(model.device)

# 1️⃣ 標準LongLLaDA生成
import time
start_time = time.time()
vanilla_outputs = model.generate(
    input_ids,
    max_new_tokens=128,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    use_cache=True
)
vanilla_time = time.time() - start_time
vanilla_result = tokenizer.decode(vanilla_outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

# 2️⃣ LongLLaDA + Fast-dLLM生成
fast_outputs, nfe, metrics = generate_fast_long(
    model=model,
    prompt=input_ids,
    gen_length=128,
    steps=128,
    block_length=32,
    temperature=0.0,
    remasking='low_confidence',
    dual_cache=True
)
fast_result = tokenizer.decode(fast_outputs[0, input_ids.shape[1]:], skip_special_tokens=True)

# 3️⃣ 結果比較
print("🐢 標準LongLLaDA:")
print(f"  結果: {vanilla_result}")
print(f"  速度: {(vanilla_outputs.shape[1] - input_ids.shape[1]) / vanilla_time:.1f} tok/s")

print("\n🚀 LongLLaDA + Fast-dLLM:")
print(f"  結果: {fast_result}")
print(f"  速度: {metrics.get('tokens_per_second', 0):.1f} tok/s")
print(f"  拡散ステップ: {nfe}")
```

---

## 🔧 長文対応（RoPE スケーリング）

### 16k トークン対応
```python
from transformers import AutoConfig

# コンフィグの修正
config = AutoConfig.from_pretrained('/content/llada-8b', trust_remote_code=True)
scaling_factor = 14  # 16k 用のスケール係数
config.rope_theta = config.rope_theta * scaling_factor

print(f"RoPE θ: {config.rope_theta} (元: 10000)")

# スケーリング適用モデルの読み込み
model_long = AutoModelForCausalLM.from_pretrained(
    '/content/llada-8b',
    config=config,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

# 長文プロンプトでテスト
long_prompt = "あなたは小説を書くAIです。" + "物語の背景設定を詳しく説明してください。" * 100  # 疑似長文
input_ids_long = tokenizer(long_prompt, return_tensors='pt').input_ids.to(model_long.device)

print(f"入力長: {input_ids_long.shape[1]} トークン")

# 長文対応生成（RoPEスケーリング + デュアルキャッシュ）
if input_ids_long.shape[1] > 4000:  # 4k を超える場合
    outputs_long, nfe, metrics = generate_fast_long_dual_cache(
        model=model_long,
        prompt=input_ids_long,
        steps=64,           # 長文では少なめに
        gen_length=512,
        block_length=64,    # 大きめのブロック
        temperature=0.0,
        remasking='low_confidence',
        scaling_factor=14   # 16k対応
    )
    result_long = tokenizer.decode(outputs_long[0, input_ids_long.shape[1]:], skip_special_tokens=True)
    print("長文生成結果:", result_long[:200] + "...")
    print(f"速度: {metrics['tokens_per_second']:.1f} tok/s")
```

---

## ⚡ 高速化オプション

### 1. ブロックサイズ調整
```python
# 高速重視（品質やや低下）
outputs_fast, nfe_fast, metrics_fast = generate_fast_long_dual_cache(
    model, input_ids,
    steps=64,           # ステップ数削減
    gen_length=256,
    block_length=64,    # 大きなブロック
    temperature=0.0,
    remasking='random'  # ランダムマスキング
)

# 品質重視（速度やや低下）
outputs_quality, nfe_quality, metrics_quality = generate_fast_long_dual_cache(
    model, input_ids,
    steps=256,          # ステップ数増加
    gen_length=256,
    block_length=16,    # 小さなブロック
    temperature=0.0,
    remasking='low_confidence'
)
```

### 2. スケール係数表
| 目標長 | スケール係数 (λ) | 備考 |
|--------|------------------|------|
| 8k     | 4                | 軽い拡張 |
| 16k    | 14               | 標準的 |
| 24k    | 31               | 大幅拡張 |
| 32k    | 55               | 最大級 |

---

## 🧪 LLM-as-a-Judge 評価システム

### 🎯 評価の核心
**標準LongLLaDA** と **LongLLaDA + Fast-dLLM** のどちらが優れているかを客観的に判定します。

### 📏 評価メトリクス
| 指標 | 説明 | 重要度 |
|------|------|--------|
| **勝率** | LLM Judge による優劣判定の勝率 | ⭐⭐⭐ |
| **生成速度** | トークン/秒の定量比較 | ⭐⭐⭐ |
| **Judge一貫性** | J-score（評価の安定性） | ⭐⭐ |
| **NIAH成功率** | 長文での情報検索能力 | ⭐⭐ |

### 🧑‍⚖️ LLM Judge 評価の実行方法
```python
# 1. 環境設定
export GOOGLE_API_KEY="your-api-key"
pip install google-generativeai

# 2. 比較評価実行
python llm_judge_evaluation.py --eval-type comprehensive

# 3. 結果確認
# comprehensive_evaluation_results.json が生成される
```

### 🔍 評価フロー詳細
1. **同一プロンプト**で両手法による生成を実行
2. **出力をシャッフル**してバイアスを除去（盲検評価）
3. **Gemini 2.0 Flash**が優劣を判定（温度=0で一貫性確保）
4. **勝率を集計**：>50% で優勢と判定
5. **J-score計算**：同一評価を複数回実行して Judge の一貫性を測定

### 📊 期待される結果
```python
# 仮想的な結果例
📊 包括的評価結果:
============================
📝 基本比較:
  標準LongLLaDA: 45.0% 勝率 (12.3 tok/s)
  LongLLaDA+Fast-dLLM: 55.0% 勝率 (18.7 tok/s)  # 🎯期待される改善
  Judge Consistency: 0.823

🔢 スケーリング比較:
  16k設定: 42.0% 勝率
  32k設定: 58.0% 勝率
```

> **🔬 研究価値**: Fast-dLLMにより品質を保ちながら**1.5倍の高速化**が実現されるかを検証

### NIAH（Needle in a Haystack）テスト
```python
def niah_test(model, tokenizer, context_length=8000):
    # ダミー長文コンテキスト生成
    haystack = "これは重要ではない情報です。" * (context_length // 10)
    needle = "重要な情報：答えは42です。"
    question = "重要な情報は何ですか？"
    
    # 長文作成
    full_text = haystack[:len(haystack)//2] + needle + haystack[len(haystack)//2:] + "\n質問：" + question
    
    input_ids = tokenizer(full_text, return_tensors='pt').input_ids
    print(f"コンテキスト長: {input_ids.shape[1]} トークン")
    
    # 生成
    outputs = generate(model, input_ids, steps=64, gen_length=50, block_length=32)
    result = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    # 評価
    success = "42" in result
    print(f"NIAH 成功: {success}")
    print(f"回答: {result}")
    return success

# テスト実行
niah_test(model_long, tokenizer)
```

---

## 🔧 トラブルシューティング

### よくある問題と解決法

1. **CUDA Out of Memory**
   ```python
   # メモリ使用量削減
   torch.cuda.empty_cache()
   # または block_length を大きくする
   outputs = generate(model, input_ids, block_length=128)
   ```

2. **生成品質の低下**
   ```python
   # steps を増やすか、temperature を調整
   outputs = generate(model, input_ids, steps=256, temperature=0.1)
   ```

3. **長文で NaN エラー**
   ```python
   # float32 を使用
   model = model.to(torch.float32)
   ```

---

## 📊 パフォーマンス目安

| 手法 | トークン/秒 | GPU メモリ | 品質 | 備考 |
|------|-------------|-----------|------|------|
| **標準LongLLaDA** | ~12-15 | 12GB | 高 | Hugging Face標準 |
| **LongLLaDA + Fast-dLLM** | ~18-25 | 14GB | 高 | 拡散生成機構 |
| **長文（16k+）** | ~8-12 | 16GB | 高 | RoPEスケーリング適用 |

※ T4 GPU での概算値。**Fast-dLLMによる1.5-2倍の高速化を期待**

---

## 📚 参考文献

- **Fast-dLLM**: [arXiv:2409.XXXXX](https://arxiv.org/abs/2409.XXXXX)
- **LongLLaDA**: [arXiv:2409.YYYYY](https://arxiv.org/abs/2409.YYYYY)
- **LLaDA**: [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

---

このREADMEはGoogle Colabでの実験を想定しています。ローカル環境での実行時は適宜パスを調整してください。 