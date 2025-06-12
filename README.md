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
- **Webアプリケーション**: https://w87iwps1jk.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **査定API**: 未デプロイ（ECRデプロイが必要）

## 概要

このアプリケーションは以下の構成になっています：

- **フロントエンド**: Django（ユーザーインターフェース）
- **バックエンド**: FastAPI（査定API）
- **機械学習**: scikit-learn（不動産価格予測モデル）
- **デプロイ**: AWS Lambda + API Gateway

## 機能

- 物件の基本情報入力（都道府県、市区町村、地区、土地面積、建物面積、築年数）
- 機械学習モデルによる査定価格予測
- 査定結果の信頼度表示
- 査定要因の分析表示

## アーキテクチャ

```
[ユーザー] 
    ↓
[Django Frontend - AWS Lambda]
    ↓ HTTP Request  
[FastAPI Backend - AWS Lambda]
    ↓
[ML Model + Mock Data]
```

## 必要な環境

- Python 3.11+
- Docker & Docker Compose（推奨）
- AWS CLI（デプロイ時）

## 📱 アプリケーションの使い方

### Webアプリケーション

1. **アクセス**: https://w87iwps1jk.execute-api.ap-northeast-1.amazonaws.com/Prod/
2. **査定開始**: 「査定を開始する」ボタンをクリック
3. **情報入力**:
   - 都道府県（例：東京都）
   - 市区町村（例：渋谷区）
   - 地区名（例：恵比寿）
   - 土地面積（㎡）
   - 建物面積（㎡）
   - 築年数（年）
4. **査定実行**: 「査定を実行」ボタンをクリック
5. **結果確認**: 査定価格、信頼度、価格帯、査定要因を確認

### API直接利用

```bash
# 注意: 現在APIは未デプロイのため、デプロイ後にURLを更新してください
curl -X POST "https://<api-id>.execute-api.ap-northeast-1.amazonaws.com/Prod/api/valuation" \
  -H "Content-Type: application/json" \
  -d '{
    "prefecture": "東京都",
    "city": "渋谷区",
    "district": "恵比寿",
    "land_area": 100.0,
    "building_area": 80.0,
    "building_age": 10
  }'
```

## ローカル環境での実行

### Docker Composeを使用（推奨）

```bash
# リポジトリをクローン
git clone <repository-url>
cd satei_app

# Docker Composeで起動
cd deployment
docker-compose up --build

# アクセス
# フロントエンド: http://localhost:8080
# API: http://localhost:8000
```

## 🚀 AWS Lambda デプロイ

### 前提条件

- AWS CLI の設定
- SAM CLI のインストール
- Docker のインストール

### デプロイ方法

#### Django フロントエンド（ZIP デプロイ）

```bash
cd deployment
sam build -t lambda-django.yml
sam deploy --template-file .aws-sam/build/template.yaml \
  --stack-name satei-django-dev \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides Environment=dev \
  ValuationApiUrl=https://<api-id>.execute-api.ap-northeast-1.amazonaws.com/Prod \
  --resolve-s3
```

#### FastAPI バックエンド（ECR デプロイ - 推奨）

MLライブラリのサイズが大きいため、ECRコンテナデプロイを使用：

```bash
cd deployment
./ecr-deploy.sh api dev
```

#### 統合デプロイスクリプト

```bash
cd deployment
# ECRでAPIをデプロイ
./deploy_unified.sh ecr api dev

# 通常のZIPデプロイ（ML依存関係なし）
./deploy_unified.sh aws django dev
```

## 📊 API仕様

### 査定API

**エンドポイント**: `POST /api/valuation`

**リクエスト**:
```json
{
  "prefecture": "東京都",
  "city": "渋谷区",
  "district": "渋谷1-1-1",
  "land_area": 100.0,
  "building_area": 80.0,
  "building_age": 10
}
```

**レスポンス**:
```json
{
  "estimated_price": 75000000,
  "confidence": 85.2,
  "price_range": {
    "min": 65000000,
    "max": 85000000
  },
  "factors": [
    "土地面積が広く、価格にプラス影響",
    "比較的新しい物件で、価格への影響は中程度"
  ]
}
```

## 📊 データソース

**現在のデータソース**:
- モックデータを使用したデモンストレーション
- 実用版では国土交通省の不動産取引価格情報API を使用予定
- API: https://www.reinfolib.mlit.go.jp/help/apiManual/

## 🤖 機械学習モデル

- **アルゴリズム**: Random Forest Regressor
- **特徴量**: 都道府県、市区町村、地区、土地面積、建物面積、築年数
- **評価指標**: MAE (Mean Absolute Error), R² Score
- **現在の状態**: モックデータでのデモンストレーション

## 免責事項

本査定結果は参考値であり、実際の不動産価値を保証するものではありません。実際の売買価格は市場状況、物件の状態、立地条件等により変動する可能性があります。正確な査定については、不動産専門業者にご相談ください。

## ライセンス

MIT License

