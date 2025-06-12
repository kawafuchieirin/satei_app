# 不動産査定アプリ

不動産の基本情報を入力して、機械学習モデルによる査定額を取得できるWebアプリケーションです。

## 🌐 デプロイ済みアプリケーション

**現在の本番環境**：
- **Webアプリケーション**: https://a2evu7tm1a.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **査定API**: https://s97f0cugki.execute-api.ap-northeast-1.amazonaws.com/Prod/

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

1. **アクセス**: https://a2evu7tm1a.execute-api.ap-northeast-1.amazonaws.com/Prod/
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
curl -X POST "https://s97f0cugki.execute-api.ap-northeast-1.amazonaws.com/Prod/api/valuation" \
  -H "Content-Type: application/json" \
  -d '{
    "prefecture": "東京都",
    "city": "渋谷区",
    "area": "恵比寿",
    "land_area": 100.0,
    "building_area": 80.0,
    "age": 10
  }'
```

## ローカル環境での実行

### Docker Composeを使用（推奨）

```bash
# リポジトリをクローン
git clone <repository-url>
cd satei_app

# Docker Composeで起動
docker-compose up --build

# アクセス
# フロントエンド: http://localhost:8080
# API: http://localhost:8000
```

### 個別に実行

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

## 🚀 AWS Lambda デプロイ

### 前提条件

- AWS CLI の設定
- SAM CLI のインストール
- Docker のインストール

### 簡単デプロイ（推奨）

```bash
# 両方同時デプロイ
./deploy_all.sh both dev

# 個別デプロイ
./deploy_all.sh django dev    # Djangoのみ
./deploy_all.sh api dev       # FastAPIのみ
```

### 手動デプロイ

```bash
# Django アプリケーション
sam build --template-file deploy/lambda-django.yml
sam deploy --template-file deploy/lambda-django.yml --stack-name satei-django

# FastAPI アプリケーション
sam build --template-file deploy/lambda-api.yml
sam deploy --template-file deploy/lambda-api.yml --stack-name satei-api
```

## 📊 API仕様

### 査定API

**エンドポイント**: `POST /api/valuation`

**リクエスト**:
```json
{
  "prefecture": "東京都",
  "city": "渋谷区",
  "area": "恵比寿",
  "land_area": 100.0,
  "building_area": 80.0,
  "age": 10
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

## 📁 ディレクトリ構造

```
satei_app/
├── 📁 satei_project/          # Django プロジェクト設定
│   ├── settings.py             # Lambda環境対応設定
│   └── urls.py                 # メインURLルーティング
├── 📁 valuation/              # Django アプリケーション
│   ├── 📁 templates/          # HTMLテンプレート
│   ├── forms.py               # フォーム定義
│   └── views.py               # ビューロジック（API連携含む）
├── 📁 api/                   # FastAPI アプリケーション
│   ├── 📁 models/            # MLモデルとデータ処理
│   ├── main.py               # FastAPI メイン
│   └── lambda_main.py        # Lambda ハンドラー
├── 📁 deploy/               # AWS デプロイ設定
│   ├── lambda-django.yml      # Django Lambda SAMテンプレート
│   └── lambda-api.yml         # FastAPI Lambda SAMテンプレート
├── 📁 scripts/              # MLモデル管理スクリプト
├── deploy_all.sh             # 統合デプロイスクリプト
├── docker-compose.yml        # Docker Compose 設定
└── requirements.txt          # Python依存関係
```

## 免責事項

本査定結果は参考値であり、実際の不動産価値を保証するものではありません。実際の売買価格は市場状況、物件の状態、立地条件等により変動する可能性があります。正確な査定については、不動産専門業者にご相談ください。

## ライセンス

MIT License

## 🛠️ 開発者向け情報

### ローカル開発コマンド

```bash
# Django開発サーバー
python manage.py runserver

# FastAPI開発サーバー
cd api && uvicorn main:app --reload

# MLモデル作成
python scripts/create_model.py

# モデル評価
python test_model_accuracy.py
```

### AWSデプロイコマンド

```bash
# 統合デプロイ
./deploy_all.sh both dev

# 個別デプロイ
./deploy_all.sh django dev
./deploy_all.sh api dev
```

### Django設定ファイルのポイント

```python
# Lambda環境でのCSRF設定
if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
    CSRF_TRUSTED_ORIGINS = ['https://*.execute-api.ap-northeast-1.amazonaws.com']
    FORCE_SCRIPT_NAME = '/Prod'  # API GatewayのProdステージ対応
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

1. **API呼び出し失敗**
   - DjangoかFastAPIへの通信エラー
   - `valuation/views.py` でAPI URLを確認
   - CSRFトークンの設定を確認

2. **"{"message":"Forbidden"}"エラー**
   - DjangoのURLルーティング問題
   - `settings.py` の `FORCE_SCRIPT_NAME` 設定を確認
   - API GatewayのProdステージパスの不一致

3. **Lambdaデプロイエラー**
   - パッケージサイズが250MBを超えた場合
   - MLライブラリを除いた軽量版requirements.txtを使用

4. **フォームデータが送信されない**
   - DjangoとFastAPI間のフィールド名不一致
   - `views.py` でフィールドマッピングを確認してください

### 現在のデプロイ状態

- ✅ Django フロントエンド: 正常動作
- ✅ FastAPI バックエンド: 正常動作
- ✅ 査定機能: モックデータで動作中
- ✅ フォーム統合: 正常動作

### サポート

問題が発生した場合は、以下の情報を含めて報告してください：
- エラーメッセージの全文
- 発生時の操作手順
- ブラウザのコンソールログ