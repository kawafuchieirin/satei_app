#!/bin/bash

# ECRへのDockerイメージデプロイスクリプト

set -e

# 引数の処理
SERVICE=${1:-api}  # api または django
ENVIRONMENT=${2:-dev}  # dev または prod

# 設定
AWS_REGION="ap-northeast-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
API_REPO_NAME="satei-api"

# ECRにログイン
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# APIサービスのECRデプロイ
if [ "$SERVICE" = "api" ]; then
    echo "Deploying FastAPI service to ECR..."
    
    # ECRリポジトリが存在しない場合は作成
    aws ecr describe-repositories --repository-names $API_REPO_NAME --region $AWS_REGION || \
        aws ecr create-repository --repository-name $API_REPO_NAME --region $AWS_REGION

    # APIアプリケーションイメージのビルド（Lambda用Dockerfile使用）
    echo "Building API application image for Lambda..."
    cd ../valuation-api
    docker build -f Dockerfile.lambda -t $API_REPO_NAME:latest .

    # タグ付け
    docker tag $API_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest

    # ECRにプッシュ
    echo "Pushing API image to ECR..."
    docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest

    # SAMでデプロイ
    cd ../deployment
    echo "Deploying Lambda function with ECR image..."
    sam build -t lambda-container.yml
    sam deploy --template-file .aws-sam/build/template.yaml \
        --stack-name satei-api-${ENVIRONMENT} \
        --capabilities CAPABILITY_IAM \
        --parameter-overrides Environment=${ENVIRONMENT} \
        ImageUri=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${API_REPO_NAME}:latest \
        --resolve-s3

    echo "FastAPI ECR deployment completed successfully!"
    echo "API image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest"
else
    echo "Invalid service: $SERVICE. Use 'api' for FastAPI deployment."
    exit 1
fi