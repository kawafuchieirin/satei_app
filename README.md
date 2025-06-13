# 不動産査定アプリケーション

不動産の基本情報を入力して、機械学習モデルによる査定額を取得できるWebアプリケーションです。

## 📁 ディレクトリ構成

プロジェクトは以下の4つの主要カテゴリに整理されています：

### 1. **deployment/** - デプロイ
AWS へのデプロイに必要なすべてのファイル
- SAM テンプレート（Lambda設定）
- デプロイスクリプト  
- Docker 設定

### 2. **valuation-api/** - 査定API
FastAPI ベースの ML バックエンドサービス
- 機械学習モデルによる価格予測
- RESTful API エンドポイント
- Lambda 対応

### 3. **valuation-app/** - 査定アプリ
Django ベースのフロントエンドアプリケーション
- ユーザーインターフェース
- フォーム処理
- API 連携

### 4. **model-creation/** - モデル作成
ML モデルの作成・管理ツール
- モデル学習スクリプト
- データ処理
- 評価ツール

## 🌐 デプロイ済みアプリケーション

**現在の本番環境**：
- **Webアプリケーション（Django）**: https://imi1rg1eyc.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **査定API（FastAPI）**: https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **査定機能**: FastAPI バックエンドサービス（ルールベース計算）

## 概要

このアプリケーションはマイクロサービスアーキテクチャで構成されています：

- **フロントエンド**: Django（ユーザーインターフェース、フォーム処理）
- **バックエンド**: FastAPI（査定APIサービス）
- **査定ロジック**: FastAPI内のLightweightModel（ルールベース計算）
- **機械学習**: scikit-learn（ローカル開発用、Lambda では軽量モデルを使用）
- **デプロイ**: AWS Lambda + API Gateway（両サービスとも Lambda で実行）

## 機能

- 物件の基本情報入力（東京都23区限定、土地面積、建物面積、築年数）
- ルールベースによる査定価格予測
- 査定結果の信頼度表示
- 査定要因の分析表示
- 価格表示形式：「X,XXX万円」形式

## アーキテクチャ

```
[ユーザー] → [Django Lambda] → HTTP POST → [FastAPI Lambda] → [Valuation Model]
                ↓                            ↓
        [Form Validation]            [LightweightModel]
                ↓                            ↓
        [Result Display] ← JSON ← [Price Calculation]
```

### サービス間通信
- Django は環境変数 `VALUATION_API_URL` で指定された FastAPI エンドポイントに HTTP POST リクエストを送信
- FastAPI は JSON レスポンスで査定結果を返却

## 必要な環境

- Python 3.11+
- Docker & Docker Compose（推奨）
- AWS CLI（デプロイ時）

## 📱 アプリケーションの使い方

### Webアプリケーション

1. **アクセス**: https://imi1rg1eyc.execute-api.ap-northeast-1.amazonaws.com/Prod/
2. **査定開始**: 「査定を開始する」ボタンをクリック
3. **情報入力**:
   - 都道府県（固定：東京都）
   - 市区町村（23区から選択、オートサジェスト機能付き）
   - 地区名（例：恵比寿）
   - 土地面積（㎡）
   - 建物面積（㎡）
   - 築年数（年）
4. **査定実行**: 「査定を実行」ボタンをクリック
5. **結果確認**: 査定価格（X,XXX万円形式）、信頼度、価格帯、査定要因を確認

### 査定機能の詳細

査定機能は FastAPI バックエンドサービスで実行されます：

- **東京23区対応**: 各区の基準価格データを FastAPI 側で管理
- **ルールベース計算**: 土地面積 × 基準価格 + 建物面積 × 減価計算（FastAPI で処理）
- **築年数減価**: 年3%の減価率（最低30%まで）
- **価格変動**: ランダム要素で±10%の変動
- **オートサジェスト**: Django フロントエンドで23区名の入力補完機能
- **モデル自動切替**: Lambda 環境では軽量モデル、ローカルでは ML モデルを自動選択

## ローカル環境での実行

### Docker Composeを使用（推奨）

```bash
# リポジトリをクローン
git clone <repository-url>
cd satei_app

# Docker Composeで起動（両サービスを同時に起動）
cd deployment
docker-compose up --build

# アクセス
# フロントエンド: http://localhost:8080
# API: http://localhost:8000
```

### 個別サービスの起動

```bash
# Django フロントエンドのみ
cd valuation-app
python manage.py runserver 0.0.0.0:8080

# FastAPI バックエンドのみ  
cd valuation-api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🚀 AWS Lambda デプロイ

### 前提条件

- AWS CLI の設定
- SAM CLI のインストール
- Docker のインストール

### デプロイ方法

#### 統合デプロイスクリプト（推奨）

```bash
cd deployment
# 両サービスを AWS にデプロイ
./deploy_unified.sh aws both prod

# API のみ ECR 経由でデプロイ（大きな依存関係がある場合）
./deploy_unified.sh ecr api prod

# ローカル Docker デプロイ
./deploy_unified.sh local both
```

#### 個別デプロイ（手動）

**重要**: FastAPI を先にデプロイし、その URL を Django デプロイ時に指定する必要があります。

1. **FastAPI デプロイ**
```bash
cd deployment
sam build -t lambda-api-light.yml
sam deploy --stack-name satei-api-light --resolve-s3
```

2. **Django デプロイ**（FastAPI の URL を指定）
```bash
sam build -t lambda-django.yml
sam deploy --stack-name satei-django-app \
  --parameter-overrides \
    Environment=prod \
    ValuationApiUrl=https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod \
  --capabilities CAPABILITY_IAM \
  --resolve-s3
```

#### 本番環境の現在の設定

- **Django スタック**: satei-django-app
- **FastAPI スタック**: satei-api-light
- **環境**: prod  
- **Django URL**: https://imi1rg1eyc.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **FastAPI URL**: https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/

## 📊 査定ロジック仕様

### 査定計算方式

**ルールベース計算**（FastAPI の LightweightModel で実装）:

1. **基準価格**: 東京23区ごとの万円/㎡単価
2. **土地価格**: 土地面積 × 基準価格
3. **建物価格**: 建物面積 × 基準価格 × 0.8 × 減価率
4. **減価率**: max(0.3, 1 - 築年数 × 0.03)
5. **最終価格**: (土地価格 + 建物価格) × 変動率(0.9-1.1)

**23区基準価格例**:
- 千代田区: 250万円/㎡
- 港区: 220万円/㎡  
- 渋谷区: 180万円/㎡
- 新宿区: 150万円/㎡

## 📊 データソース

**現在のデータソース**:
- 東京23区の基準価格データ（静的データ）
- ルールベース計算ロジック
- 実用版では国土交通省の不動産取引価格情報API を使用予定
- API: https://www.reinfolib.mlit.go.jp/help/apiManual/

## 🧪 テスト

### Django テスト
```bash
cd valuation-app
python manage.py test valuation
```

### API エンドポイントテスト
```bash
# 本番環境の API テスト
curl -X POST https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"渋谷区","district":"恵比寿","land_area":100,"building_area":80,"building_age":10}'

# ローカル API テスト
curl http://localhost:8000/api/valuation -X POST -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"港区","district":"六本木","land_area":150,"building_area":120,"building_age":5}'
```

## 🤖 機械学習モデル

### モデル作成と管理

```bash
# 新しいモデルの作成と学習
cd model-creation
python create_model.py --data-source sample --model-type rf

# モデルパフォーマンスの評価
python model-creation/test_model_accuracy.py

# クイックモデル作成（デフォルト設定を使用）
python model-creation/quick_model.py
```

### モデルアーキテクチャ
- **アルゴリズム**: Random Forest Regressor
- **特徴量**: 都道府県、市区町村、地区、土地面積、建物面積、築年数
- **評価指標**: MAE (Mean Absolute Error), R² Score
- **デプロイ戦略**: 
  - ローカル開発: フル ML モデル（scikit-learn）
  - Lambda 環境: 軽量ルールベースモデル（依存関係制限のため）

## 免責事項

本査定結果は参考値であり、実際の不動産価値を保証するものではありません。実際の売買価格は市場状況、物件の状態、立地条件等により変動する可能性があります。正確な査定については、不動産専門業者にご相談ください。

## ライセンス

MIT License

