# 🐳 Docker Compose 開発環境

## 📋 概要

ローカル開発環境でもFastAPIベースのML査定APIを動作させ、本番環境とのロジック差分をゼロにするためのDocker Compose環境です。

## 🏗️ アーキテクチャ

```
┌─────────────────┐    HTTP     ┌─────────────────┐
│  Django App     │ ──────────→ │  FastAPI App    │
│  (Port 8000)    │             │  (Port 8001)    │
│  - UI管理       │             │  - ML査定API    │
│  - フォーム     │             │  - 共通モジュール │
└─────────────────┘             └─────────────────┘
         │                               │
         └───────── valuation_core ──────┘
                   (共通MLロジック)
```

## 📂 ディレクトリ構成

```
satei_app/
├── docker-compose.yml          # マルチコンテナ定義
├── .env                        # 環境変数設定
├── valuation_core/             # 共通MLモジュール
│   ├── __init__.py
│   └── ml_predictor.py         # 統一ML予測ロジック
├── django_app/                 # Django UI アプリ
│   ├── Dockerfile
│   ├── requirements.txt
│   └── (Django app files)
├── fastapi_app/                # FastAPI 査定 API
│   ├── Dockerfile
│   ├── main_shared.py          # 共通モジュール使用版
│   ├── requirements-ml.txt
│   └── (FastAPI app files)
└── model-creation/models/      # MLモデルファイル
    ├── valuation_model.joblib  # 最適化Random Forest
    ├── label_encoders.joblib   # カテゴリエンコーダー
    └── scaler.joblib          # 特徴量スケーラー
```

## 🚀 使用方法

### 1. 起動

```bash
# Docker Composeでビルド・起動
docker-compose up --build -d

# ログ確認
docker-compose logs -f
```

### 2. アクセス

- **Django UI**: http://localhost:8000
- **FastAPI API**: http://localhost:8001
- **API Doc**: http://localhost:8001/docs

### 3. ヘルスチェック

```bash
# FastAPI ヘルスチェック
curl http://localhost:8001/health

# Django経由でのAPI接続確認
curl http://localhost:8000/test-api/
```

### 4. 査定テスト

```bash
# FastAPI直接テスト
curl -X POST http://localhost:8001/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"渋谷区","land_area":100,"building_area":80,"building_age":10}'

# Django UIテスト
# http://localhost:8000 でフォーム入力
```

### 5. 停止・クリーンアップ

```bash
# 停止
docker-compose down

# ボリューム含めて完全削除
docker-compose down -v

# イメージも削除
docker-compose down --rmi all
```

## 🔧 環境設定

### .env ファイル

```bash
# Django Settings
DEBUG=True
VALUATION_API_URL=http://fastapi:8001

# FastAPI Settings
MODEL_PATH=/app/models
ENABLE_ML=true
LOG_LEVEL=INFO
```

### ポート設定

- Django: 8000
- FastAPI: 8001

ポート競合時は `docker-compose.yml` で変更してください。

## 📊 MLモデル詳細

### 共通MLPredictor

- **モデル**: 最適化Random Forest (4.1MB)
- **性能**: R² = 0.8066
- **訓練データ**: 63,217件の東京23区実取引データ
- **特徴量**: 5次元 (市区町村、地区、土地面積、建物面積、築年数)

### 予測例

```json
{
  "estimated_price": 62014158.73,
  "confidence": 87.0,
  "price_range": {
    "min": 52712034.92,
    "max": 71316282.54
  },
  "factors": [
    "機械学習モデルによる高精度予測",
    "最適化Random Forest アルゴリズム使用",
    "63,217件の実取引データで訓練"
  ]
}
```

## 🐛 トラブルシューティング

### ポート競合

```bash
# 使用中ポートの確認
lsof -i :8000
lsof -i :8001

# プロセス強制停止
kill -9 <PID>
```

### モデルファイル問題

```bash
# ボリュームマウント確認
docker-compose exec fastapi ls -la /app/models/

# ログ確認
docker-compose logs fastapi | grep -i model
```

### Django API接続問題

```bash
# ネットワーク確認
docker-compose exec django ping fastapi

# 環境変数確認
docker-compose exec django env | grep VALUATION_API_URL
```

## ✨ 利点

1. **本番環境同等**: AWS Lambda環境と同じMLロジック
2. **開発効率**: コード変更の即座反映 (--reload)
3. **スケーラビリティ**: マイクロサービス構成
4. **共通モジュール**: Django/FastAPI間のロジック統一
5. **モデル共有**: 同一MLモデルファイルの使用

## 📈 次のステップ

- [ ] Kubernetes対応
- [ ] CI/CD パイプライン統合
- [ ] モニタリング・ログ集約
- [ ] API認証機能
- [ ] ヘルスチェック機能拡張