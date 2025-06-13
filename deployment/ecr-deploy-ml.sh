#!/bin/bash

# ECRへのMLモデル付きDockerイメージデプロイスクリプト

set -e

# 引数の処理
SERVICE=${1:-api}  # api のみサポート
ENVIRONMENT=${2:-dev}  # dev または prod

# 設定
AWS_REGION="ap-northeast-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
API_REPO_NAME="satei-api-ml"

# ECRにログイン
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# APIサービスのECRデプロイ
if [ "$SERVICE" = "api" ]; then
    echo "Deploying ML-enabled FastAPI service to ECR..."
    
    # ECRリポジトリが存在しない場合は作成
    aws ecr describe-repositories --repository-names $API_REPO_NAME --region $AWS_REGION || \
        aws ecr create-repository --repository-name $API_REPO_NAME --region $AWS_REGION

    # APIアプリケーションイメージのビルド（ML依存関係を含む）
    echo "Building ML API application image for Lambda..."
    cd ../valuation-api-ml
    
    # ML対応のDockerfileを作成
    cat > Dockerfile.lambda.ml << 'EOF'
FROM public.ecr.aws/lambda/python:3.9

# ML依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . ${LAMBDA_TASK_ROOT}

# モデルファイルディレクトリの作成
RUN mkdir -p ${LAMBDA_TASK_ROOT}/models

# モデルクリエーションディレクトリのコピー（必要な場合）
COPY ../model-creation ${LAMBDA_TASK_ROOT}/model-creation

# ハンドラーの設定
CMD ["lambda_main.handler"]
EOF

    docker build -f Dockerfile.lambda.ml -t $API_REPO_NAME:latest .

    # タグ付け
    docker tag $API_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest

    # ECRにプッシュ
    echo "Pushing ML API image to ECR..."
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest

    # SAMでデプロイ
    cd ../deployment
    echo "Deploying ML Lambda function with ECR image..."
    
    # ML用のSAMテンプレートを使用
    sam build -t lambda-container-ml.yml
    sam deploy --template-file .aws-sam/build/template.yaml \
        --stack-name satei-api-ml-${ENVIRONMENT} \
        --capabilities CAPABILITY_IAM \
        --parameter-overrides Environment=${ENVIRONMENT} \
        ImageUri=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${API_REPO_NAME}:latest \
        --resolve-s3

    echo "ML-enabled FastAPI ECR deployment completed successfully!"
    echo "ML API image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest"
    
    # デプロイ後の情報を表示
    echo ""
    echo "Deployment Information:"
    echo "======================"
    API_URL=$(aws cloudformation describe-stacks \
        --stack-name satei-api-ml-${ENVIRONMENT} \
        --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
        --output text)
    echo "API URL: $API_URL"
    
else
    echo "Invalid service: $SERVICE. Use 'api' for ML FastAPI deployment."
    exit 1
fi