# 機械学習実装ガイド

## 概要

本ドキュメントは、不動産査定アプリケーションのルールベースロジックから機械学習モデルへの移行について説明します。

## アーキテクチャの変更点

### 以前の構成（ルールベース）
- 東京23区の固定基準価格を使用
- 築年数による一律3%/年の減価償却
- ±10%のランダム変動

### 新しい構成（機械学習）
- MLIT API（国土交通省）の実データを使用
- 複数のMLモデル（線形回帰、Ridge、Lasso、Random Forest、XGBoost）
- データドリブンな価格予測

## ディレクトリ構成

```
satei_app/
├── valuation-api-ml/        # ML対応API
│   ├── main_ml.py          # ML専用APIエンドポイント
│   ├── models/
│   │   └── valuation_model.py  # ML専用モデル（ルールベース削除済）
│   └── requirements-ml.txt  # ML依存関係
├── model-creation/          # モデル訓練・管理
│   ├── tokyo23_data_fetcher.py    # MLIT APIデータ取得
│   ├── data_preprocessor.py       # データ前処理
│   ├── train_xgboost_model.py     # マルチモデル訓練
│   └── model_evaluator.py         # モデル評価
└── deployment/              # デプロイメント設定
    ├── ecr-deploy-ml.sh           # ML版ECRデプロイ
    └── lambda-container-ml.yml    # ML版SAMテンプレート
```

## セットアップ手順

### 1. 環境準備

```bash
# Python環境のセットアップ
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# ML依存関係のインストール
pip install -r valuation-api-ml/requirements-ml.txt
pip install -r model-creation/requirements.txt
```

### 2. データ取得

```bash
cd model-creation

# MLIT APIからデータを取得（2022-2024年）
python tokyo23_data_fetcher.py

# 取得したデータは data/ ディレクトリに保存されます
```

### 3. モデル訓練

```bash
# 全モデルの訓練と比較
python train_xgboost_model.py

# 訓練結果：
# - models/best_model.joblib      # 最良モデル
# - models/model_comparison.png   # モデル比較グラフ
# - models/model_comparison_results.json  # 詳細結果
```

### 4. モデル評価

```bash
# モデルの詳細評価
python model_evaluator.py data/tokyo23_real_estate_*.csv

# 評価結果：
# - models/evaluation_report.json
# - models/model_comparison_visualization.png
```

## API使用方法

### ローカル開発

```bash
cd valuation-api-ml
uvicorn main_ml:app --reload --port 8000
```

### APIエンドポイント

#### 1. 価格査定
```bash
curl -X POST http://localhost:8000/api/valuation \
  -H "Content-Type: application/json" \
  -d '{
    "prefecture": "東京都",
    "city": "港区",
    "district": "六本木",
    "land_area": 150,
    "building_area": 120,
    "building_age": 5
  }'
```

レスポンス例：
```json
{
  "estimated_price": 185000000,
  "confidence": 90.0,
  "price_range": {
    "min": 166500000,
    "max": 203500000
  },
  "model_type": "xgboost",
  "model_metrics": {
    "r2": 0.89,
    "rmse": 12500000
  }
}
```

#### 2. モデル訓練
```bash
curl -X POST http://localhost:8000/api/model/train \
  -H "Content-Type: application/json" \
  -d '{
    "fetch_new_data": false,
    "test_size": 0.2,
    "cv_folds": 5
  }'
```

#### 3. モデル情報取得
```bash
curl http://localhost:8000/api/model/info
```

#### 4. MLITデータ取得
```bash
curl "http://localhost:8000/api/data/fetch?from_year=2022&to_year=2024"
```

## デプロイメント

### ECRを使用したLambdaデプロイ

```bash
cd deployment

# ML版APIのECRデプロイ
./ecr-deploy-ml.sh api prod

# デプロイ後、API URLが表示されます
```

### Docker Composeでの開発

```bash
cd deployment
docker-compose up --build
```

## モデルの詳細

### 使用可能なモデル

1. **線形回帰（Linear Regression）**
   - シンプルで解釈しやすい
   - ベースラインモデル

2. **Ridge回帰**
   - L2正則化付き線形回帰
   - 過学習を防ぐ

3. **Lasso回帰**
   - L1正則化付き線形回帰
   - 特徴選択効果

4. **Random Forest**
   - アンサンブル学習
   - 非線形関係を捉える

5. **XGBoost**
   - 勾配ブースティング
   - 最高精度（通常）

### 特徴量

- 土地面積（㎡）
- 建物面積（㎡）
- 築年数
- 市区町村（エンコード済）
- 地区名（エンコード済）
- 建ぺい率
- 容積率
- 前面道路幅員
- 最寄駅距離

## トラブルシューティング

### モデルファイルが見つからない

```bash
# モデルの再訓練
cd model-creation
python train_xgboost_model.py
```

### メモリ不足エラー

```bash
# Lambda設定でメモリを増やす
# lambda-container-ml.yml の MemorySize: 3008 に設定
```

### MLIT APIエラー

```bash
# APIキーが必要な場合
export MLIT_API_KEY=your_api_key
python tokyo23_data_fetcher.py
```

## パフォーマンス指標

典型的なモデル性能（XGBoost）：
- R²スコア: 0.85-0.92
- RMSE: 1,000万-1,500万円
- MAPE: 10-15%

## 今後の改善案

1. **特徴量エンジニアリング**
   - 駅からの距離データの追加
   - 周辺施設情報の統合

2. **モデルの改善**
   - ディープラーニングモデルの検討
   - 時系列要素の追加

3. **運用の改善**
   - モデルの定期的な再訓練
   - A/Bテストの実装
   - モニタリングの強化