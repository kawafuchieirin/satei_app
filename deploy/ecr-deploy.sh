#!/bin/bash

# ECRへのDockerイメージデプロイスクリプト

set -e

# 設定
AWS_REGION="ap-northeast-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
DJANGO_REPO_NAME="satei-django"
API_REPO_NAME="satei-api"

# ECRにログイン
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# ECRリポジトリが存在しない場合は作成
aws ecr describe-repositories --repository-names $DJANGO_REPO_NAME --region $AWS_REGION || \
    aws ecr create-repository --repository-name $DJANGO_REPO_NAME --region $AWS_REGION

aws ecr describe-repositories --repository-names $API_REPO_NAME --region $AWS_REGION || \
    aws ecr create-repository --repository-name $API_REPO_NAME --region $AWS_REGION

# Djangoアプリケーションイメージのビルド
echo "Building Django application image..."
docker build -t $DJANGO_REPO_NAME:latest .

# APIアプリケーションイメージのビルド
echo "Building API application image..."
docker build -t $API_REPO_NAME:latest ./api

# タグ付け
docker tag $DJANGO_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DJANGO_REPO_NAME:latest
docker tag $API_REPO_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest

# ECRにプッシュ
echo "Pushing Django image to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DJANGO_REPO_NAME:latest

echo "Pushing API image to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest

echo "Deploy completed successfully!"
echo "Django image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DJANGO_REPO_NAME:latest"
echo "API image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$API_REPO_NAME:latest"