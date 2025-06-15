# デプロイメント

不動産査定アプリケーションのデプロイメント設定が役割別に整理されています。

## 📁 ディレクトリ構成

```
deployment/
├── aws/          # AWS本番デプロイ用ファイル
│   ├── README.md
│   ├── deploy_all.sh
│   ├── ecr-deploy-ml.sh
│   └── lambda-*.yml
├── local/        # ローカル開発環境用ファイル
│   ├── README.md
│   └── docker-compose.yml
└── README.md     # このファイル
```

## 🚀 クイックスタート

### ローカル開発環境
```bash
cd deployment/local
docker-compose up --build -d
```

### AWS本番デプロイ
```bash
cd deployment/aws
./deploy_all.sh both prod
```

## 📖 詳細ドキュメント

- **ローカル開発**: [local/README.md](local/README.md)
- **AWS本番デプロイ**: [aws/README.md](aws/README.md)

## 🎯 環境の統一

この構成により以下が実現できます：

- **開発環境**: Docker Composeで本番同等のMLロジック
- **本番環境**: AWS Lambdaで高スケーラブルな運用
- **統一性**: 両環境で同一のMLモデルと予測ロジック

## 🔄 移行ガイド

従来の使用方法からの変更点：

**従来**: `./deploy_all.sh both prod`  
**新方式**: `cd deployment/aws && ./deploy_all.sh both prod`

**従来**: `docker-compose up`  
**新方式**: `cd deployment/local && docker-compose up`