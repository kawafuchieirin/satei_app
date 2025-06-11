# 不動産査定アプリ

不動産の基本情報を入力して、機械学習モデルによる査定額を取得できるWebアプリケーションです。

## 概要

このアプリケーションは以下の構成になっています：

- **フロントエンド**: Django（ユーザーインターフェース）
- **バックエンド**: FastAPI（査定API）
- **機械学習**: scikit-learn（不動産価格予測モデル）
- **データソース**: 国土交通省 不動産取引価格情報API

## 機能

- 物件の基本情報入力（所在地、面積、築年数など）
- 機械学習モデルによる査定価格予測
- 査定結果の信頼度表示
- 査定要因の分析表示

## アーキテクチャ

```
[ユーザー] 
    ↓
[Django Frontend (port 8080)]
    ↓ HTTP Request
[FastAPI Backend (port 8000)]
    ↓
[ML Model + MLIT Data]
```

## 必要な環境

- Python 3.11+
- Docker & Docker Compose（推奨）
- AWS CLI（デプロイ時）

## ローカル環境での実行

### 1. Docker Composeを使用（推奨）

```bash
# リポジトリをクローン
git clone <repository-url>
cd satei_app

# Docker Composeで起動
docker-compose up --build

# アクセス
# フロントエンド: http://localhost:3000
# API: http://localhost:3001
```

### 2. 個別に実行

#### Django フロントエンド

```bash
# 依存関係のインストール
pip install -r requirements.txt

# データベースマイグレーション
python manage.py migrate

# 開発サーバー起動
python manage.py runserver 0.0.0.0:8080
```

#### FastAPI バックエンド

```bash
cd api

# 依存関係のインストール
pip install -r requirements.txt

# APIサーバー起動
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 設定

### 環境変数

`.env` ファイルを作成して以下の設定を行ってください：

```env
DEBUG=True
SECRET_KEY=your-secret-key
VALUATION_API_URL=http://localhost:8000
```

## AWS Lambda デプロイ

### 前提条件

- AWS CLI の設定
- SAM CLI のインストール
- ECR リポジトリの作成権限

### ECR デプロイ

```bash
# ECRにDockerイメージをデプロイ
./deploy/ecr-deploy.sh
```

### Lambda関数デプロイ

```bash
# Django アプリケーション
sam deploy --template-file deploy/lambda-django.yml --stack-name satei-django

# FastAPI アプリケーション
sam deploy --template-file deploy/lambda-api.yml --stack-name satei-api
```

## API仕様

### 査定API

**エンドポイント**: `POST /api/valuation`

**リクエスト**:
```json
{
  "prefecture": "東京都",
  "city": "渋谷区",
  "district": "恵比寿",
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

## データソース

本アプリケーションは国土交通省の不動産取引価格情報を使用しています：
- API: https://www.reinfolib.mlit.go.jp/help/apiManual/
- データ期間: 直近3年間の取引データ

## 機械学習モデル

- **アルゴリズム**: Random Forest Regressor
- **特徴量**: 都道府県、市区町村、地区、土地面積、建物面積、築年数
- **評価指標**: MAE (Mean Absolute Error), R² Score

## ディレクトリ構造

```
satei_app/
├── satei_project/          # Django プロジェクト設定
├── valuation/              # Django アプリケーション
│   ├── templates/          # HTMLテンプレート
│   ├── forms.py           # フォーム定義
│   └── views.py           # ビュー関数
├── api/                   # FastAPI アプリケーション
│   ├── models/            # MLモデルとデータ処理
│   ├── main.py           # FastAPI メイン
│   └── lambda_main.py    # Lambda ハンドラー
├── lambda/               # Lambda 設定
├── deploy/               # デプロイ設定
├── docker-compose.yml    # Docker Compose 設定
└── README.md
```

## 免責事項

本査定結果は参考値であり、実際の不動産価値を保証するものではありません。実際の売買価格は市場状況、物件の状態、立地条件等により変動する可能性があります。正確な査定については、不動産専門業者にご相談ください。

## ライセンス

MIT License

## 開発者向け情報

### テスト実行

```bash
# Django テスト
python manage.py test

# FastAPI テスト
cd api
pytest
```

### コード品質チェック

```bash
# flake8
flake8 .

# black (コードフォーマット)
black .
```

## トラブルシューティング

### よくある問題

1. **APIとの通信エラー**
   - FastAPIサーバーが起動しているか確認
   - ファイアウォール設定を確認

2. **モデル学習エラー**
   - MLIT APIへのアクセスができない場合、サンプルデータが使用されます
   - インターネット接続を確認

3. **Docker起動エラー**
   - Dockerが起動しているか確認
   - ポート8000, 8080が使用されていないか確認

### サポート

問題が発生した場合は、GitHubのIssuesページで報告してください。