# モデル作成スクリプト集

不動産査定AIモデルを作成・管理するためのスクリプト集です。

## 📁 ファイル構成

```
scripts/
├── model_manager.py          # 🎯 統合管理スクリプト（推奨）
├── quick_model.py           # ⚡ クイックモデル作成
├── create_model.py          # 🔬 詳細モデル作成
├── batch_model_training.py  # 🚀 バッチ学習
└── README.md               # 📖 このファイル
```

## 🎯 統合管理スクリプト（推奨）

全機能を統合した使いやすいメインスクリプトです。

### 基本的な使用方法

```bash
# 現在の状態確認
python scripts/model_manager.py status

# クイックモデル作成（推奨）
python scripts/model_manager.py quick

# モデル評価
python scripts/model_manager.py evaluate

# モデル比較
python scripts/model_manager.py compare

# モデルデプロイ
python scripts/model_manager.py deploy
```

### 利用可能なコマンド

| コマンド | 説明 | 所要時間 | 推奨度 |
|----------|------|----------|--------|
| `status` | 現在のモデル状態を表示 | 数秒 | ★★★ |
| `quick` | 高速でバランスの良いモデルを作成 | 1-3分 | ★★★ |
| `create` | 詳細設定で高精度モデルを作成 | 5-15分 | ★★ |
| `batch` | 複数モデルを並列学習して最適化 | 10-30分 | ★★ |
| `evaluate` | 現在のモデルを詳細評価 | 1-2分 | ★★★ |
| `compare` | 過去のモデルと比較 | 数秒 | ★★ |
| `deploy` | モデルをAPIに反映 | 数秒 | ★★★ |

## ⚡ クイックモデル作成

最も簡単で高速なモデル作成方法です。**初心者にオススメ**。

### 使用方法

```bash
# バランス型（推奨）
python scripts/quick_model.py --preset balanced

# 高速型（精度やや劣る、速度重視）
python scripts/quick_model.py --preset fast

# 高精度型（時間かかる、最高精度）
python scripts/quick_model.py --preset best
```

### 特徴

- **所要時間**: 1-3分
- **難易度**: ★☆☆
- **精度**: ★★☆ - ★★★
- **設定**: プリセットのみ
- **推奨用途**: 初回作成、プロトタイピング

## 🔬 詳細モデル作成

高度な設定が可能な本格的なモデル作成ツールです。

### 使用方法

```bash
# 基本的な詳細作成
python scripts/create_model.py --data-source sample

# ハイパーパラメータ調整なし（高速）
python scripts/create_model.py --data-source sample --no-grid-search

# MLIT実データを使用（APIキー必要）
python scripts/create_model.py --data-source mlit

# 交差検証の分割数変更
python scripts/create_model.py --cv-folds 10

# 評価も同時実行
python scripts/create_model.py --data-source sample --evaluate
```

### オプション

- `--data-source`: データソース（sample/mlit/csv）
- `--no-grid-search`: ハイパーパラメータ調整スキップ
- `--cv-folds`: 交差検証分割数（デフォルト: 5）
- `--test-size`: テストデータ割合（デフォルト: 0.2）
- `--output-dir`: 出力ディレクトリ
- `--evaluate`: 学習後に評価実行

### 特徴

- **所要時間**: 5-15分
- **難易度**: ★★☆
- **精度**: ★★★
- **設定**: 詳細設定可能
- **推奨用途**: 本格運用、精度重視

## 🚀 バッチ学習

複数の設定でモデルを並列学習し、最適なモデルを自動選択します。

### 使用方法

```bash
# 基本的なバッチ学習
python scripts/batch_model_training.py --data-source sample

# 並列数指定
python scripts/batch_model_training.py --max-workers 4

# MLIT実データ使用
python scripts/batch_model_training.py --data-source mlit
```

### 学習されるモデル

1. **Random Forest** (4種類の設定)
2. **Gradient Boosting** (3種類の設定)
3. **Ridge回帰** (4種類のα値)
4. **Lasso回帰** (3種類のα値)

### 特徴

- **所要時間**: 10-30分
- **難易度**: ★★★
- **精度**: ★★★
- **設定**: 自動最適化
- **推奨用途**: 最高精度が必要、時間に余裕がある場合

## 📊 実行例

### 1. 初回セットアップ

```bash
# 1. 現在の状態確認
python scripts/model_manager.py status

# 2. クイックモデル作成
python scripts/model_manager.py quick --preset balanced

# 3. モデル評価
python scripts/model_manager.py evaluate

# 4. APIサーバーに反映
python scripts/model_manager.py deploy
```

### 2. 精度改善

```bash
# 1. 詳細モデル作成
python scripts/model_manager.py create --data-source sample

# 2. バッチ学習で最適化
python scripts/model_manager.py batch --max-workers 4

# 3. モデル比較
python scripts/model_manager.py compare

# 4. 最良モデルをデプロイ
python scripts/model_manager.py deploy
```

### 3. 運用・監視

```bash
# 定期的な状態確認
python scripts/model_manager.py status

# 新しいデータでモデル更新
python scripts/model_manager.py quick --preset best

# 評価・比較
python scripts/model_manager.py evaluate
python scripts/model_manager.py compare
```

## 🎯 推奨ワークフロー

### 初心者向け

1. `python scripts/model_manager.py quick` - クイックモデル作成
2. `python scripts/model_manager.py evaluate` - 評価確認
3. `python scripts/model_manager.py deploy` - デプロイ

### 上級者向け

1. `python scripts/model_manager.py create` - 詳細モデル作成
2. `python scripts/model_manager.py batch` - バッチ最適化
3. `python scripts/model_manager.py compare` - 比較分析
4. `python scripts/model_manager.py deploy` - 最良モデルデプロイ

## 📈 出力ファイル

### モデルファイル（本番用）

- `../api/valuation_model.joblib` - メインモデル
- `../api/label_encoders.joblib` - カテゴリ変数エンコーダー

### 学習情報ファイル

- `quick_model_info_YYYYMMDD_HHMMSS.json` - クイック学習情報
- `training_info_YYYYMMDD_HHMMSS.json` - 詳細学習情報
- `batch_training_results_YYYYMMDD_HHMMSS.json` - バッチ学習結果
- `model_report_YYYYMMDD_HHMMSS.md` - 詳細レポート

### 可視化ファイル（詳細作成時）

- `model_comparison.png` - モデル比較グラフ
- `best_model_analysis.png` - 最良モデル分析
- `feature_importance.png` - 特徴量重要度

## ⚠️ 注意事項

### システム要件

- Python 3.11+
- 必要なライブラリ: pandas, scikit-learn, matplotlib, seaborn
- メモリ: 最低2GB（バッチ学習時は4GB推奨）
- CPU: マルチコア推奨（並列処理のため）

### データソースについて

- **sample**: 人工データ、常に利用可能、開発・テスト用
- **mlit**: 国土交通省実データ、APIキー必要、本番用
- **csv**: カスタムデータ、`training_data.csv`が必要

### パフォーマンス目安

| スクリプト | データ件数 | 実行時間 | メモリ使用量 |
|------------|------------|----------|--------------|
| quick_model | 1,500件 | 1-3分 | 500MB |
| create_model | 2,000件 | 5-15分 | 1-2GB |
| batch_training | 2,000件 | 10-30分 | 2-4GB |

## 🔧 トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   ```bash
   # データサイズを削減
   python scripts/quick_model.py --preset fast
   ```

2. **MLIT API接続エラー**
   ```bash
   # サンプルデータに切り替え
   python scripts/model_manager.py quick --preset balanced
   ```

3. **モデルが読み込まれない**
   ```bash
   # APIサーバー再起動
   docker-compose restart api
   ```

4. **スクリプト実行エラー**
   ```bash
   # 権限付与
   chmod +x scripts/*.py
   
   # 依存関係インストール
   pip install pandas scikit-learn matplotlib seaborn
   ```

### ログ確認

スクリプト実行時のログは標準出力に表示されます。エラーが発生した場合は、エラーメッセージを確認してください。

### サポート

問題が解決しない場合は、以下の情報と共にサポートまでお問い合わせください：

1. 実行したコマンド
2. エラーメッセージ
3. システム情報（OS、Python版数）
4. データサイズ・メモリ使用量

## 📚 関連ドキュメント

- [MODEL_CREATION_GUIDE.md](../MODEL_CREATION_GUIDE.md) - 詳細な作成ガイド
- [MODEL_VALIDATION_REPORT.md](../MODEL_VALIDATION_REPORT.md) - 精度検証レポート
- [README.md](../README.md) - プロジェクト全体のREADME