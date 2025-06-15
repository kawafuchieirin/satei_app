# ローカル開発環境

## Docker Compose 統合開発環境

このディレクトリには、ローカル開発用のDocker Compose設定が含まれています。

### 🚀 クイックスタート

```bash
# 開発環境の起動
cd deployment/local
docker-compose up --build -d

# アクセス
# Django UI: http://localhost:8000
# FastAPI API: http://localhost:8001/docs

# ログ確認
docker-compose logs -f

# 停止
docker-compose down
```

### 🔧 サービス構成

- **Django** (Port 8000): ユーザーインターフェース
- **FastAPI** (Port 8001): ML査定API
- **共通モジュール**: valuation_core (統一MLロジック)

### 📊 開発中のテスト

```bash
# 査定API直接テスト
curl -X POST http://localhost:8001/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"渋谷区","land_area":100,"building_area":80,"building_age":10}'

# Django経由のE2Eテスト
curl http://localhost:8000/test-api/

# ヘルスチェック
curl http://localhost:8001/health
```

### 🎯 本番同等環境

この環境は本番AWSと同じMLロジックを使用しており、開発・テストが本番と同等の結果を提供します。