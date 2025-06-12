# 不動産査定アプリ - プロジェクト構造

## 📁 ディレクトリ構造

```
satei_app/
├── 📁 api/                        # FastAPI バックエンド
│   ├── 📁 models/                 # MLモデルとデータ処理
│   │   ├── data_fetcher.py        # 国土交通省APIからのデータ取得
│   │   ├── model_evaluator.py     # モデル評価機能
│   │   └── valuation_model.py     # 不動産査定MLモデル
│   ├── main.py                    # FastAPIメインアプリケーション
│   ├── lambda_main.py             # AWS Lambda用ハンドラー
│   └── requirements.txt           # Python依存関係
│
├── 📁 valuation/                  # Django フロントエンド
│   ├── 📁 templates/              # HTMLテンプレート
│   │   └── 📁 valuation/
│   │       ├── base.html          # ベーステンプレート
│   │       ├── index.html         # ホームページ
│   │       ├── form.html          # 査定フォーム
│   │       └── result.html        # 査定結果表示
│   ├── views.py                   # ビューロジック
│   ├── forms.py                   # フォーム定義
│   └── urls.py                    # URLルーティング
│
├── 📁 satei_project/              # Django プロジェクト設定
│   ├── settings.py                # アプリケーション設定
│   └── urls.py                    # メインURLルーティング
│
├── 📁 deploy/                     # AWS デプロイ設定
│   ├── lambda-django.yml          # Django Lambda SAMテンプレート
│   ├── lambda-api.yml             # FastAPI Lambda SAMテンプレート
│   └── deploy.sh                  # 汎用デプロイスクリプト
│
├── 📁 scripts/                    # MLモデル管理スクリプト
│   ├── create_model.py            # モデル作成
│   ├── model_manager.py           # モデル管理
│   └── batch_model_training.py    # バッチ学習
│
└── 📁 lambda/                     # Lambda関連ファイル
    └── django-lambda.py           # Django Lambda設定
```

## 🚀 主要ファイル

### デプロイメント
- `deploy_all.sh` - 統合デプロイスクリプト（Django + API）
- `samconfig.toml` - AWS SAM設定ファイル
- `docker-compose.yml` - ローカル開発用Docker設定

### アプリケーション設定
- `manage.py` - Djangoコマンドラインツール
- `requirements.txt` - メインPython依存関係
- `.gitignore` - Git除外設定

### ドキュメント
- `README.md` - プロジェクト概要
- `MODEL_CREATION_GUIDE.md` - MLモデル作成ガイド
- `MODEL_VALIDATION_REPORT.md` - モデル検証レポート

## 🏗️ アーキテクチャ

### フロントエンド (Django)
- **URL**: https://a2evu7tm1a.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **機能**: Webインターフェース、フォーム処理
- **デプロイ**: AWS Lambda + API Gateway

### バックエンド (FastAPI)
- **URL**: https://s97f0cugki.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **機能**: 不動産査定API、MLモデル実行
- **デプロイ**: AWS Lambda + API Gateway

### データフロー
```
ユーザー → Django (フォーム) → FastAPI (査定) → Django (結果表示)
```

## 🛠️ 開発・デプロイコマンド

### ローカル開発
```bash
# Django開発サーバー
python manage.py runserver

# FastAPI開発サーバー
cd api && uvicorn main:app --reload
```

### AWS Lambdaデプロイ
```bash
# 両方同時デプロイ
./deploy_all.sh both dev

# 個別デプロイ
./deploy_all.sh django dev
./deploy_all.sh api dev
```

### MLモデル管理
```bash
# モデル作成
python scripts/create_model.py

# モデル評価
python test_model_accuracy.py
```

## 📊 現在のデプロイ状況

- ✅ Django フロントエンド: デプロイ済み、正常動作
- ✅ FastAPI バックエンド: デプロイ済み、正常動作  
- ✅ 査定機能: 完全動作（モック査定）
- ✅ フォーム統合: 正常動作
- ✅ URL ルーティング: 修正済み