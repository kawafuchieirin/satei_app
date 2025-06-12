# デプロイ

このディレクトリにはAWSへのデプロイに必要なすべてのファイルが含まれています。

## 主要ファイル
- **lambda-django.yml**: Django Lambda デプロイ用 SAM テンプレート
- **lambda-api.yml**: FastAPI Lambda デプロイ用 SAM テンプレート
- **lambda-container.yml**: ECR コンテナデプロイ用 SAM テンプレート
- **ecr-deploy.sh**: ECR デプロイスクリプト
- **deploy_unified.sh**: 統合デプロイスクリプト
- **docker-compose.yml**: ローカル開発用 Docker 設定

## デプロイ方法

### Django デプロイ
```bash
sam build -t lambda-django.yml
sam deploy --template-file .aws-sam/build/template.yaml --stack-name satei-django-dev --capabilities CAPABILITY_IAM --resolve-s3
```

### FastAPI デプロイ (ECR)
```bash
./ecr-deploy.sh api dev
```
