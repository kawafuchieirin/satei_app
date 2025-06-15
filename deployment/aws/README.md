# AWS本番デプロイメント

## AWS Lambda デプロイ設定

このディレクトリには、AWS環境への本番デプロイ用ファイルが含まれています。

### 🚀 統合デプロイ (推奨)

```bash
# 両サービスを同時デプロイ
./deploy_all.sh both prod

# APIのみデプロイ
./deploy_all.sh api prod

# Djangoのみデプロイ
./deploy_all.sh django prod
```

### 🔧 ML対応ECRデプロイ (本格運用)

```bash
# ML完全版APIをECRコンテナでデプロイ
./ecr-deploy-ml.sh api prod
# - 3GB RAM, 5分タイムアウト
# - 118.8MB訓練済みモデル搭載
# - XGBoost, Random Forest対応
```

### 📁 ファイル構成

- `deploy_all.sh`: 統合デプロイスクリプト
- `ecr-deploy-ml.sh`: ECRベースMLデプロイ
- `lambda-api.yml`: FastAPI用SAMテンプレート
- `lambda-django.yml`: Django用SAMテンプレート
- `lambda-container-ml.yml`: ML対応ECR用SAMテンプレート

### 🎯 本番URL

- **Django**: https://imi1rg1eyc.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **FastAPI (ML版)**: https://25cfdqih7a.execute-api.ap-northeast-1.amazonaws.com/Prod/

### 📊 デプロイ後テスト

```bash
# 本番API疎通確認
curl -X POST https://25cfdqih7a.execute-api.ap-northeast-1.amazonaws.com/Prod/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"港区","district":"六本木","land_area":150,"building_area":120,"building_age":5}'
```