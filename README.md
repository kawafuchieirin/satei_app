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
- **Webアプリケーション**: https://imi1rg1eyc.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **査定機能**: Django内蔵（直接計算方式）

## 概要

このアプリケーションは以下の構成になっています：

- **フロントエンド**: Django（ユーザーインターフェース）
- **査定ロジック**: Django内蔵（ルールベース計算）
- **機械学習**: scikit-learn（将来の拡張用）
- **デプロイ**: AWS Lambda + API Gateway

## 機能

- 物件の基本情報入力（東京都23区限定、土地面積、建物面積、築年数）
- ルールベースによる査定価格予測
- 査定結果の信頼度表示
- 査定要因の分析表示
- 価格表示形式：「X,XXX万円」形式

## アーキテクチャ

```
[ユーザー] 
    ↓
[Django Frontend - AWS Lambda]
    ↓ 内部関数呼び出し
[ルールベース査定ロジック]
    ↓
[東京23区価格データ + 計算式]
```

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

現在の査定機能は Django アプリケーション内で直接実行されます：

- **東京23区対応**: 各区の基準価格データを内蔵
- **ルールベース計算**: 土地面積 × 基準価格 + 建物面積 × 減価計算
- **築年数減価**: 年3%の減価率（最低30%まで）
- **価格変動**: ランダム要素で±10%の変動
- **オートサジェスト**: 23区名の入力補完機能

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

#### Django アプリケーション（本体デプロイ）

```bash
cd deployment
sam build -t lambda-django.yml
sam deploy --stack-name satei-django-app \
  --parameter-overrides Environment=prod \
  --capabilities CAPABILITY_IAM \
  --resolve-s3
```

#### 本番環境の現在の設定

- **スタック名**: satei-django-app
- **環境**: prod  
- **URL**: https://imi1rg1eyc.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **査定機能**: Django 内蔵（FastAPI 不要）

## 📊 査定ロジック仕様

### 査定計算方式

**ルールベース計算**（Django 内蔵）:

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

## 🤖 機械学習モデル（将来拡張用）

- **アルゴリズム**: Random Forest Regressor
- **特徴量**: 都道府県、市区町村、地区、土地面積、建物面積、築年数
- **評価指標**: MAE (Mean Absolute Error), R² Score
- **現在の状態**: ルールベース計算で代替実装

## 免責事項

本査定結果は参考値であり、実際の不動産価値を保証するものではありません。実際の売買価格は市場状況、物件の状態、立地条件等により変動する可能性があります。正確な査定については、不動産専門業者にご相談ください。

## ライセンス

MIT License

