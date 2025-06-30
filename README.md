# Fast-dLLM × LongLLaDA 統合実験（Google Colab 対応）

Fast-dLLM の高速推論機構と LongLLaDA の長文拡張機構を **LLaDA** で組み合わせるための手順書です。Google Colab での実行に最適化されています。

## 📋 概要

| 機能 | 実装箇所 | 効果 |
|------|----------|------|
| **ブロック生成** | `LongLLaDA/llada/llada_generate.py` | 並列トークン生成で高速化 |
| **RoPE スケーリング** | `LongLLaDA/llada/llada_wrapper.py` | 長文コンテキスト対応 |
| **信頼度制御** | 生成関数の `remasking='low_confidence'` | 品質維持 |

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

## 💡 基本的な使用方法

### 標準生成（4k コンテキスト）
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from LongLLaDA.llada.llada_generate import generate

# モデル・トークナイザの読み込み
model = AutoModelForCausalLM.from_pretrained(
    '/content/llada-8b',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    '/content/llada-8b', 
    trust_remote_code=True
)

# プロンプトの準備
prompt = "次の数学問題を解いてください：太郎は毎時12kmで4時間走り、その後毎時6kmで走りました。合計8時間でどれだけの距離を走れますか？"

# チャットテンプレートの適用
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=False
)

input_ids = tokenizer(formatted_prompt, return_tensors='pt').input_ids.to(model.device)

# 生成実行
outputs = generate(
    model=model,
    prompt=input_ids,
    steps=128,          # 拡散ステップ数
    gen_length=256,     # 生成長
    block_length=32,    # ブロックサイズ
    temperature=0.0,    # 決定的生成
    remasking='low_confidence'
)

# 結果の表示
result = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
print("生成結果:", result)
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

# 長文対応生成
if input_ids_long.shape[1] > 4000:  # 4k を超える場合
    outputs_long = generate(
        model=model_long,
        prompt=input_ids_long,
        steps=64,           # 長文では少なめに
        gen_length=512,
        block_length=64,    # 大きめのブロック
        temperature=0.0,
        remasking='low_confidence'
    )
    result_long = tokenizer.decode(outputs_long[0, input_ids_long.shape[1]:], skip_special_tokens=True)
    print("長文生成結果:", result_long[:200] + "...")
```

---

## ⚡ 高速化オプション

### 1. ブロックサイズ調整
```python
# 高速重視（品質やや低下）
outputs_fast = generate(
    model, input_ids,
    steps=64,           # ステップ数削減
    gen_length=256,
    block_length=64,    # 大きなブロック
    temperature=0.0,
    remasking='random'  # ランダムマスキング
)

# 品質重視（速度やや低下）
outputs_quality = generate(
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

## 🧪 評価・テスト
### 🗒️ 標準ベンチマーク
- **MMLU**: 一般知識の多肢選択問題
- **GSM8K**: 小学校レベル算数
- **HumanEval**: プログラム合成
- **LongBench**: 長文理解＆検索
- **NIAH**: Needle-in-a-Haystack（後述）

### 📏 評価メトリクス
| カテゴリ | 指標 | 説明 |
|----------|------|------|
| 自動評価 | Accuracy / Pass@k / EM | タスク固有の定量指標 |
| スピード | tok/s | 生成トークン数 ÷ 時間 |
| **LLM Judge** | Pairwise Preference | GPT-4 による出力ペア比較 |
| **Judge Consistency** | J-score | 生成の一貫性を GPT で採点 |

### 🧑‍⚖️ LLM-as-a-Judge 評価フロー
1. 2 つのモデル出力を **シャッフルして非公開 ID** を付与  
2. **Gemini 2.0 Flash** に **プロンプト・出力ペア** を渡し「どちらが優れているか」 を尋ねる  
3. ①で付けた ID を元に勝率を集計 → **勝率 >50% → 優勢**  
4. 追加で同一回答セットを複数回評価し **J-score** を計算し Judge の一貫性を確認

```python
# Colab: Google API キーを環境変数 GOOGLE_API_KEY に設定済みと仮定
!pip install google-genai tqdm
import os, json, random, tqdm
from google import genai

client = genai.Client()  # GOOGLE_API_KEY を自動取得

def llm_judge(prompts, outputs_a, outputs_b, judge_model="gemini-2.0-flash"):
    """
    Gemini 2.0 Flash を用いたペアワイズ評価
    prompts:   List[str]
    outputs_a: List[str]  # モデルAの出力
    outputs_b: List[str]  # モデルBの出力
    戻り値:    dict(score=float, details=list)
    """
    wins = 0
    details = []
    for p, a, b in tqdm.tqdm(zip(prompts, outputs_a, outputs_b), total=len(prompts)):
        # 出力をランダムで並べ替えてバイアスを除去
        pair = list(zip(["A","B"], [a, b]))
        random.shuffle(pair)
        labels, answers = zip(*pair)

        prompt_text = (
            "## プロンプト\n" + p +
            "\n\n### 回答A\n" + answers[0] +
            "\n\n### 回答B\n" + answers[1] +
            "\n\n# 指示\n"
            "優れている方のラベル ('A' or 'B') のみを出力してください。"
        )

        resp = client.models.generate_content(
            model=judge_model,
            contents=prompt_text,
            config={"temperature":0.0}
        )
        choice = resp.text.strip()
        preferred = labels[0] if choice=="A" else labels[1]
        wins += (preferred=="A")  # A が勝った回数をカウント
        details.append(preferred)
    return {"score":wins/len(prompts), "details":details}

上記の関数で **score > 0.5** ならモデルA がモデルB を上回っていると判断します。

> 💡 **TIP**: Judge の温度を 0 に固定し、同一プロンプトを複数回評価して安定性を確認するとより信頼性が高まります。

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

| 設定 | トークン/秒 | GPU メモリ | 品質 |
|------|-------------|-----------|------|
| 標準 | ~80 | 12GB | 高 |
| 高速 | ~120 | 10GB | 中 |
| 長文 | ~60 | 16GB | 高 |

※ T4 GPU での概算値

---

## 📚 参考文献

- **Fast-dLLM**: [arXiv:2409.XXXXX](https://arxiv.org/abs/2409.XXXXX)
- **LongLLaDA**: [arXiv:2409.YYYYY](https://arxiv.org/abs/2409.YYYYY)
- **LLaDA**: [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

---

このREADMEはGoogle Colabでの実験を想定しています。ローカル環境での実行時は適宜パスを調整してください。 