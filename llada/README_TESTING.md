# Adaptive Scheduling Testing Guide

このガイドでは、アップデートされたテストフレームワークの使用方法を説明します。新機能には**信頼度プロット**と**パラメータ設定可能化**が含まれます。

## 🔧 新機能

### 1. 信頼度動きのプロット
- 生成中の信頼度変化をリアルタイムで可視化
- モード切り替えポイントをハイライト
- 設定可能な閾値線表示
- 統計情報の自動計算

### 2. コマンドライン引数でのパラメータ設定
- すべての主要パラメータがコマンドラインで設定可能
- デフォルト値はResearchPaperの推奨値
- 複数設定の比較機能

### 3. パラメータ設定比較
- 異なる設定の性能を並行比較
- 統合されたプロット表示
- 定量的比較結果

## 📊 使用例

### 基本的なクイックテスト（デフォルト設定）
```bash
# 複雑な推論タスクで信頼度プロットを生成
python test_adaptive_scheduling.py --quick-test complex_reasoning
```

### カスタム設定でのテスト
```bash
# より積極的な設定（低い閾値、小さいウィンドウ）
python test_adaptive_scheduling.py --quick-test complex_reasoning \
    --to-quality-threshold 0.75 \
    --to-efficiency-threshold 0.90 \
    --confidence-window-size 1 \
    --efficiency-block-size 16

# より保守的な設定（高い閾値、大きいウィンドウ）
python test_adaptive_scheduling.py --quick-test math_reasoning \
    --to-quality-threshold 0.85 \
    --to-efficiency-threshold 0.98 \
    --confidence-window-size 3
```

### パラメータ設定比較
```bash
# 複数の設定を同時に比較（デフォルト、積極的、保守的）
python test_adaptive_scheduling.py --compare-settings complex_reasoning

# 数学推論タスクでの設定比較
python test_adaptive_scheduling.py --compare-settings math_reasoning
```

### プロット無効化
```bash
# プロットを生成せずに高速実行
python test_adaptive_scheduling.py --quick-test simple_qa --no-plot
```

### 包括的評価（カスタム設定）
```bash
# 全評価をカスタム設定で実行
python test_adaptive_scheduling.py --comprehensive \
    --to-quality-threshold 0.75 \
    --efficiency-block-size 24 \
    --quality-block-size 6
```

## 🎛️ パラメータ詳細

### スケジューラー設定
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--to-quality-threshold` | 0.80 | 品質モードに切り替える信頼度閾値 |
| `--to-efficiency-threshold` | 0.95 | 効率モードに切り替える信頼度閾値 |
| `--confidence-window-size` | 2 | 信頼度平滑化ウィンドウサイズ |

### 効率モードパラメータ
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--efficiency-block-size` | 32 | 効率モードのブロックサイズ |
| `--efficiency-threshold` | 0.75 | 効率モードの信頼度閾値 |

### 品質モードパラメータ
| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--quality-block-size` | 8 | 品質モードのブロックサイズ |
| `--quality-threshold` | 0.95 | 品質モードの信頼度閾値 |

## 📈 プロット解釈

### 信頼度プロット（上段）
- **青い線**: 生成中の信頼度変化
- **赤い破線**: 品質モード閾値（これを下回ると品質モードに切り替え）
- **緑の破線**: 効率モード閾値（これを上回ると効率モードに切り替え）
- **オレンジの縦線**: モード変更ポイント

### モード切り替えプロット（下段）
- **緑のバー**: 効率モード（HIGH_EFFICIENCY）
- **赤のバー**: 品質モード（HIGH_QUALITY）
- 連続したバーがモードの持続期間を示す

### 統計情報
```
📈 信頼度統計:
   平均信頼度: 0.847
   最小信頼度: 0.623
   最大信頼度: 0.981
   信頼度標準偏差: 0.089
   モード変更回数: 3
   効率モード時間: 45/64 (70.3%)
   品質モード時間: 19/64 (29.7%)
```

## 🔍 推奨設定

### タスク別推奨設定

#### 数学推論タスク
```bash
# 精度重視
python test_adaptive_scheduling.py --quick-test math_reasoning \
    --to-quality-threshold 0.85 \
    --quality-block-size 6
```

#### コード生成
```bash
# バランス重視
python test_adaptive_scheduling.py --quick-test code_generation \
    --to-quality-threshold 0.80 \
    --efficiency-block-size 24
```

#### 創作文章
```bash
# 創造性重視
python test_adaptive_scheduling.py --quick-test creative_writing \
    --to-quality-threshold 0.75 \
    --confidence-window-size 1
```

#### 簡単なQ&A
```bash
# 効率重視
python test_adaptive_scheduling.py --quick-test simple_qa \
    --to-efficiency-threshold 0.90 \
    --efficiency-block-size 48
```

## 📁 出力ファイル

### プロット保存
- 場所: `confidence_plots/`
- 形式: `confidence_{test_case_name}_{timestamp}.png`
- 比較プロット: `comparison_{test_case_name}_{timestamp}.png`

### 結果データ
- 場所: `adaptive_scheduling_results/`
- 詳細結果: `detailed_results_{timestamp}.json`
- 比較分析: `comparison_{timestamp}.json`

## 🚀 ベストプラクティス

### 1. パラメータ探索
まず設定比較を実行して最適な設定を見つける：
```bash
python test_adaptive_scheduling.py --compare-settings your_task
```

### 2. 詳細分析
興味深い設定で詳細テストを実行：
```bash
python test_adaptive_scheduling.py --quick-test your_task \
    --your-optimal-parameters
```

### 3. 包括的評価
最終的な評価で全タスクをテスト：
```bash
python test_adaptive_scheduling.py --comprehensive \
    --your-optimal-parameters
```

## 🔧 トラブルシューティング

### プロットが表示されない
- `matplotlib`がインストールされているか確認
- `--no-plot`フラグが設定されていないか確認
- ディスプレイ設定を確認（SSH接続時など）

### パフォーマンスが遅い
- `--no-plot`を使用してプロット生成を無効化
- より小さい`--gen-length`を使用
- GPU使用可能か確認

### メモリ不足
- より小さいモデルを使用
- `--gen-length`を削減
- バッチサイズを調整

## 📝 カスタマイズ

独自のテストケースを追加する場合は、`_prepare_test_cases()`メソッドを編集してください：

```python
{
    "name": "your_custom_task",
    "prompt": "Your test prompt here",
    "category": "custom",
    "expected_difficulty": "medium",
    "expected_mode": "HIGH_QUALITY"  # オプショナル
}
``` 