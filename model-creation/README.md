# モデル作成ツール

このディレクトリには、不動産査定モデルを作成・管理するためのツールが含まれています。

## 🚀 新機能: LightGBMベースのMLモデル

東京23区の実データを使用したLightGBMモデルによる高精度な価格予測が可能になりました。

### クイックスタート

```bash
# 完全なMLパイプラインを実行（データ取得→訓練→デプロイ）
python ml_model_manager.py full

# または個別に実行
python ml_model_manager.py fetch    # データ取得
python ml_model_manager.py train    # モデル訓練
python ml_model_manager.py deploy   # デプロイ
python ml_model_manager.py evaluate # 評価
```

## 新しいMLモデルシステム

### 1. tokyo23_data_fetcher.py
国土交通省APIから東京23区の不動産取引データを取得

```bash
# 環境変数にAPIキーを設定
export MLIT_API_KEY="your_api_key_here"

# データ取得（スタンドアロン実行）
python tokyo23_data_fetcher.py
```

特徴:
- 2022-2024年の東京23区全データを自動収集
- 和暦→西暦変換、データクリーニング機能
- CSV形式で保存（`data/`ディレクトリ）

### 2. data_preprocessor.py
MLモデル用のデータ前処理

特徴:
- カテゴリカル変数のエンコーディング
- 数値データの正規化
- 特徴量エンジニアリング

### 3. train_ml_model.py
LightGBMを使用した機械学習モデルの訓練

```bash
# 新規データで訓練
python train_ml_model.py --fetch-new

# 既存CSVデータで訓練
python train_ml_model.py --data-path data/tokyo23_real_estate_2022_2024.csv

# クロスバリデーション設定
python train_ml_model.py --cv-folds 10 --test-size 0.3
```

特徴:
- LightGBM（勾配ブースティング）による高精度予測
- クロスバリデーション
- 特徴量重要度分析
- 予測信頼区間の推定

### 4. ml_model_manager.py
MLモデルの統合管理ツール

```bash
# 使用例
python ml_model_manager.py -h  # ヘルプ表示

# コマンド一覧
python ml_model_manager.py full      # 完全パイプライン
python ml_model_manager.py fetch     # データ取得のみ
python ml_model_manager.py train     # モデル訓練のみ
python ml_model_manager.py deploy    # デプロイのみ
python ml_model_manager.py evaluate  # 評価のみ
```

## FastAPI統合

### 新しいエンドポイント

valuation-api/main_ml.py で以下のエンドポイントが利用可能:

- `POST /api/valuation` - 価格予測（ML/ルールベース自動切替）
- `GET /api/model/info` - モデル情報取得
- `POST /api/model/train` - モデル訓練（MLモデルのみ）
- `GET /api/model/feature-importance` - 特徴量重要度

### 使用方法

```python
# FastAPIサーバー起動（MLモデル対応版）
cd valuation-api
uvicorn main_ml:app --reload
```

## 既存ツールとの統合

既存のツール（quick_model.py、create_model.py等）も引き続き使用可能です:

### 1. model_manager.py
統一されたモデル管理インターフェース

```bash
python model_manager.py status      # モデルの状態確認
python model_manager.py quick       # クイックモデル作成
python model_manager.py create      # 詳細モデル作成
python model_manager.py batch       # バッチモデル訓練
```

### 2. quick_model.py
プリセットを使用した高速モデル作成（RandomForest）

```bash
python quick_model.py --preset balanced
```

### 3. create_model.py
詳細な設定でモデルを作成（複数アルゴリズム対応）

```bash
python create_model.py --model-type rf --tune
```

## データソース

### MLIT API（推奨）
- 国土交通省の実データ
- APIキーが必要（[登録はこちら](https://www.reinfolib.mlit.go.jp/register.html)）
- 高精度な予測が可能

### サンプルデータ
- APIキー不要
- テスト・開発用途

## 出力ファイル

### MLモデル（LightGBM）
- `models/real_estate_model.joblib` - 訓練済みLightGBMモデル
- `models/label_encoders.joblib` - カテゴリ変数エンコーダー
- `models/scaler.joblib` - 数値正規化スケーラー
- `models/feature_columns.joblib` - 特徴量リスト
- `models/model_info.json` - モデルメタデータ
- `models/model_evaluation.png` - 評価グラフ

### レガシーモデル（RandomForest等）
- `../api/valuation_model.joblib` - 訓練済みモデル
- `../api/label_encoders.joblib` - エンコーダー
- `../api/training_info_*.json` - 訓練情報

## ワークフロー例

### 本番環境向けMLモデル構築

```bash
# 1. APIキーを設定
export MLIT_API_KEY="your_api_key"

# 2. 完全パイプライン実行
python ml_model_manager.py full

# 3. FastAPIサーバー起動
cd ../valuation-api
uvicorn main_ml:app --host 0.0.0.0 --port 8000
```

### カスタムデータでの訓練

```bash
# 1. データ準備（CSVファイル）
# 必要列: 区, 地区名, 土地面積, 建物面積, 築年数, 取引価格（総額）

# 2. モデル訓練
python train_ml_model.py --data-path your_data.csv

# 3. デプロイ
python ml_model_manager.py deploy --target api
```

## Lambda デプロイメント

LightGBMはサイズが大きいため、Lambda環境では自動的にルールベースモデルにフォールバックします。

完全なMLモデルを使用する場合:
- ECRコンテナデプロイメントを使用
- または、推論専用の軽量化されたモデルを作成

## トラブルシューティング

### LightGBMインストールエラー
```bash
# macOS
brew install libomp

# Ubuntu/Debian  
sudo apt-get install -y libgomp1
```

### メモリ不足
- データサンプリングを使用
- `--test-size` を増やして訓練データを削減

### APIキーエラー
- 環境変数 `MLIT_API_KEY` が設定されているか確認
- APIキーの有効性を確認

## 注意事項

- MLモデルは定期的な再訓練が必要です（月1回推奨）
- 実データ使用時はAPIレート制限に注意
- モデルファイルは数MB〜数十MBになる可能性があります
- Lambda環境ではファイルサイズ制限により自動的にフォールバックモデルを使用

---

## 📁 ファイル構成（既存ツール）

```
model-creation/
├── model_manager.py          # 🎯 統合管理スクリプト（推奨）
├── quick_model.py           # ⚡ クイックモデル作成
├── create_model.py          # 🔬 詳細モデル作成
├── batch_model_training.py  # 🚀 バッチ学習
├── tokyo23_data_fetcher.py  # 🆕 MLIT APIデータ取得
├── data_preprocessor.py     # 🆕 データ前処理
├── train_ml_model.py        # 🆕 LightGBMモデル訓練
├── ml_model_manager.py      # 🆕 MLモデル統合管理
└── README.md               # 📖 このファイル
```

## 🎯 統合管理スクリプト（既存）

全機能を統合した使いやすいメインスクリプトです。

### 基本的な使用方法

```bash
# 現在の状態確認
python model_manager.py status

# クイックモデル作成（推奨）
python model_manager.py quick

# モデル評価
python model_manager.py evaluate

# モデル比較
python model_manager.py compare

# モデルデプロイ
python model_manager.py deploy
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

## 📚 関連ドキュメント

- [MODEL_CREATION_GUIDE.md](../MODEL_CREATION_GUIDE.md) - 詳細な作成ガイド
- [MODEL_VALIDATION_REPORT.md](../MODEL_VALIDATION_REPORT.md) - 精度検証レポート
- [README.md](../README.md) - プロジェクト全体のREADME