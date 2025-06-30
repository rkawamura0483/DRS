# 🚀 Fast-dLLM × LongLLaDA Google Colab 実行ガイド

Fast-dLLMの高速推論機構とLongLLaDAの長文拡張機構を組み合わせた統合実験をGoogle Colabで実行するための完全ガイドです。

## 📋 事前準備

### 1. Google Colab セットアップ
- [Google Colab](https://colab.research.google.com/) にアクセス
- **GPU** または **TPU** ランタイムを選択
  - `ランタイム` → `ランタイムのタイプを変更` → `ハードウェア アクセラレータ: GPU`

### 2. Google API キー取得（LLM Judge評価用）
- [Google AI Studio](https://aistudio.google.com/) でAPIキーを取得
- 環境変数として設定

## 🔧 Google Colab での実行手順

### Step 1: リポジトリのクローンと環境セットアップ

```python
# GPU確認
!nvidia-smi

# リポジトリクローン
!git clone https://github.com/your-repo/Fast-dLLM.git
%cd Fast-dLLM

# 依存関係インストール
!pip install -r requirements.txt

# Flash Attention を個別インストール（エラーが出る場合）
!pip install flash-attn==2.3.3 --no-build-isolation
```

### Step 2: クイックスタート実行

```python
# 基本的なデモの実行
!python quick_start.py
```

このコマンドで以下が実行されます：
- ✅ 環境チェック
- 🚀 基本生成デモ（4k コンテキスト）
- 📚 長文コンテキストデモ（RoPE スケーリング）
- ⚡ 速度比較（異なる設定）
- 🎯 簡易NIAH テスト

### Step 3: 詳細評価実験

```python
# 包括的評価実験
!python run_integrated_experiment.py

# 個別テスト実行
!python run_integrated_experiment.py --test basic          # 基本生成のみ
!python run_integrated_experiment.py --test long           # 長文テストのみ
!python run_integrated_experiment.py --test niah           # NIAH テストのみ
!python run_integrated_experiment.py --test performance    # 性能比較のみ
```

### Step 4: LLM-as-a-Judge 評価

```python
# Google API キー設定
import os
os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY_HERE'

# LLM Judge 評価実行
!python llm_judge_evaluation.py
```

## 📊 実行コマンド一覧

| コマンド | 説明 | 実行時間目安 |
|----------|------|-------------|
| `python quick_start.py` | クイックデモ（推奨） | 5-10分 |
| `python run_integrated_experiment.py` | 包括的評価 | 15-30分 |
| `python run_integrated_experiment.py --test basic` | 基本テストのみ | 3-5分 |
| `python run_integrated_experiment.py --test long` | 長文テストのみ | 5-10分 |
| `python run_integrated_experiment.py --test niah` | NIAH テストのみ | 5-8分 |
| `python llm_judge_evaluation.py` | LLM Judge 評価 | 10-20分 |

## ⚙️ 詳細設定

### RoPE スケーリング係数

| 目標長 | スケーリング係数 | コマンド例 |
|--------|-----------------|-----------|
| 4k (標準) | 1 | `--scaling-factor 1` |
| 8k | 4 | `--scaling-factor 4` |
| 16k | 14 | `--scaling-factor 14` |
| 24k | 31 | `--scaling-factor 31` |
| 32k | 55 | `--scaling-factor 55` |

### 生成設定

| 用途 | steps | block_length | remasking | 特徴 |
|------|-------|--------------|-----------|------|
| 🏃 超高速 | 32 | 64 | random | 最速、品質やや低 |
| ⚡ 高速 | 64 | 32 | random | 高速、品質中 |
| 🎯 標準 | 128 | 32 | low_confidence | バランス良好 |
| 🎨 品質重視 | 256 | 16 | low_confidence | 高品質、やや低速 |

## 🔧 カスタム実行例

### 例1: 高速16k生成

```python
!python run_integrated_experiment.py \
  --test long \
  --scaling-factor 14 \
  --model GSAI-ML/LLaDA-8B-Instruct
```

### 例2: Python APIでの直接実行

```python
import sys
sys.path.append('.')

from integrated_generation import generate_fast_long, load_model_with_scaling

# モデル読み込み（16k対応）
model, tokenizer, config = load_model_with_scaling(
    'GSAI-ML/LLaDA-8B-Instruct', 
    scaling_factor=14
)

# プロンプト準備
prompt = "あなたの質問をここに入力"
messages = [{"role": "user", "content": prompt}]
formatted_prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
input_ids = tokenizer(formatted_prompt, return_tensors='pt').input_ids.to(model.device)

# 生成実行
outputs, nfe, metrics = generate_fast_long(
    model=model,
    prompt=input_ids,
    steps=128,           # 拡散ステップ数
    gen_length=256,      # 生成長
    block_length=32,     # ブロックサイズ
    temperature=0.0,     # 決定的生成
    remasking='low_confidence',  # 信頼度ベース
    use_cache=True,      # KVキャッシュ使用
    scaling_factor=14    # 16k対応
)

# 結果表示
result = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
print("生成結果:", result)
print(f"速度: {metrics['tokens_per_second']:.1f} tok/s")
```

## 📈 期待される結果

### 性能目安（T4 GPU）

| 設定 | 速度 (tok/s) | GPU メモリ | 品質 |
|------|-------------|-----------|------|
| 標準（4k） | ~80 | 12GB | 高 |
| 高速（4k） | ~120 | 10GB | 中 |
| 長文（16k） | ~60 | 16GB | 高 |

### 実装された機能

- ✅ **Fast-dLLM ブロック生成**: 並列トークン生成で高速化
- ✅ **LongLLaDA RoPE スケーリング**: 長文コンテキスト対応
- ✅ **信頼度制御**: 品質維持のための適応的マスク解除
- ✅ **KV キャッシュ最適化**: メモリ効率向上

## 🔧 トラブルシューティング

### よくある問題と解決法

#### 1. GPU メモリ不足
```
CUDA out of memory
```
**解決法:**
```python
# ブロックサイズを大きくしてメモリ使用量削減
!python run_integrated_experiment.py --test basic
# または
torch.cuda.empty_cache()  # メモリクリア
```

#### 2. Flash Attention インストールエラー
```
Failed building wheel for flash-attn
```
**解決法:**
```python
# より安定した方法でインストール
!pip install flash-attn==2.3.3 --no-build-isolation --no-cache-dir
# または Flash Attention なしで実行
```

#### 3. モデルダウンロードエラー
```
Cannot connect to Hugging Face
```
**解決法:**
```python
# Hugging Face Hub にログイン
from huggingface_hub import notebook_login
notebook_login()

# または直接トークン設定
import os
os.environ['HF_TOKEN'] = 'your_hf_token_here'
```

#### 4. 生成品質の低下
**解決法:**
```python
# ステップ数を増やす
!python run_integrated_experiment.py --test basic
# または信頼度ベースリマスキングを使用
```

#### 5. 長文で NaN エラー
**解決法:**
```python
# float32 を使用
model = model.to(torch.float32)
# またはスケーリング係数を調整
```

## 📂 出力ファイル

実行後、以下のファイルが生成されます：

- `experiment_results.json`: 詳細評価結果
- `llm_judge_results.json`: LLM Judge 評価結果
- ログファイル（コンソール出力）

## 🎯 評価メトリクス

### 自動評価
- **速度**: tokens/second
- **効率**: NFE (Number of Function Evaluations)
- **メモリ**: GPU使用量
- **精度**: NIAH 成功率

### LLM-as-a-Judge 評価
- **品質比較**: 異なる設定での生成品質
- **勝率**: モデル間の相対評価
- **一貫性**: Judge の判定安定性

## 🚀 次のステップ

1. **基本実行**: `python quick_start.py` でまず動作確認
2. **詳細評価**: `python run_integrated_experiment.py` で包括評価
3. **カスタマイズ**: `integrated_generation.py` を編集して独自実験
4. **LLM Judge**: API キー設定後に品質評価実行

## 📚 参考資料

- **Fast-dLLM**: [arXiv:2409.XXXXX](https://arxiv.org/abs/2409.XXXXX)
- **LongLLaDA**: [arXiv:2409.YYYYY](https://arxiv.org/abs/2409.YYYYY)
- **LLaDA**: [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA)

---

このガイドに従って実行することで、Fast-dLLMとLongLLaDAの統合された高速長文生成システムを体験できます。 