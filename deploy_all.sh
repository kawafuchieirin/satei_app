#!/bin/bash

# 不動産査定アプリ 統合Lambdaデプロイスクリプト
# 使用方法: ./deploy_all.sh [APP_TYPE] [ENVIRONMENT]
# APP_TYPE: django | api | both
# ENVIRONMENT: dev | prod

set -e

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${PURPLE} 🚀 不動産査定アプリ 統合Lambdaデプロイ${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

usage() {
    echo "使用方法: $0 [APP_TYPE] [ENVIRONMENT]"
    echo ""
    echo "APP_TYPE:"
    echo "  django    - Djangoフロントエンドアプリのみ"
    echo "  api       - FastAPI バックエンドのみ"
    echo "  both      - Django + FastAPI の同時デプロイ（デフォルト）"
    echo ""
    echo "ENVIRONMENT:"
    echo "  dev       - 開発環境（デフォルト）"
    echo "  prod      - 本番環境"
    echo ""
    echo "例:"
    echo "  $0                     # Django + API 開発環境同時デプロイ"
    echo "  $0 both prod          # Django + API 本番環境同時デプロイ"
    echo "  $0 django dev         # Django開発環境のみデプロイ"
    echo "  $0 api prod           # API本番環境のみデプロイ"
    echo ""
    echo "🎯 推奨: 'both' オプションで両方同時にデプロイすることをお勧めします"
}

# パラメータの設定
APP_TYPE=${1:-"both"}
ENVIRONMENT=${2:-"dev"}

# パラメータ検証
if [[ "$APP_TYPE" != "django" && "$APP_TYPE" != "api" && "$APP_TYPE" != "both" ]]; then
    echo -e "${RED}エラー: APP_TYPEは 'django', 'api', または 'both' である必要があります${NC}"
    usage
    exit 1
fi

if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "prod" ]]; then
    echo -e "${RED}エラー: ENVIRONMENTは 'dev' または 'prod' である必要があります${NC}"
    usage
    exit 1
fi

print_header

echo -e "${YELLOW}デプロイ設定:${NC}"
echo "  アプリタイプ: $APP_TYPE"
echo "  環境: $ENVIRONMENT"
echo ""

# AWS CLIとSAM CLIの確認
echo -e "${YELLOW}前提条件チェック中...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLIがインストールされていません${NC}"
    echo "インストール方法: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
    exit 1
fi

if ! command -v sam &> /dev/null; then
    echo -e "${RED}❌ SAM CLIがインストールされていません${NC}"
    echo "インストール方法: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html"
    exit 1
fi

# AWS認証確認
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS認証が設定されていません${NC}"
    echo "aws configure を実行してください"
    exit 1
fi

echo -e "${GREEN}✅ 前提条件OK${NC}"
echo ""

# デプロイ関数
deploy_django() {
    local env=$1
    echo -e "${CYAN}📦 Django アプリケーションをデプロイ中...${NC}"
    
    # API Gateway URL を設定（APIを先にデプロイする場合）
    API_URL="http://localhost:8000"
    if [[ "$APP_TYPE" == "both" ]] || [[ -n "$API_GATEWAY_URL" ]]; then
        API_URL="${API_GATEWAY_URL:-https://api-placeholder.execute-api.ap-northeast-1.amazonaws.com}"
    fi
    
    sam build -t lambda-django.yml
    sam deploy \
        --template-file .aws-sam/build/template.yaml \
        --stack-name "satei-django-${env}" \
        --capabilities CAPABILITY_IAM \
        --resolve-s3 \
        --parameter-overrides "Environment=${env}" "ValuationApiUrl=${API_URL}" \
        --no-confirm-changeset
    
    # Django API Gateway URLを取得
    DJANGO_URL=$(aws cloudformation describe-stacks \
        --stack-name "satei-django-${env}" \
        --query 'Stacks[0].Outputs[?OutputKey==`DjangoApi`].OutputValue' \
        --output text 2>/dev/null || echo "")
    
    if [[ -n "$DJANGO_URL" ]]; then
        echo -e "${GREEN}✅ Django デプロイ完了${NC}"
        echo -e "${CYAN}📍 Django URL: ${DJANGO_URL}${NC}"
    else
        echo -e "${YELLOW}⚠️  Django URLの取得に失敗しました${NC}"
    fi
}

deploy_api() {
    local env=$1
    echo -e "${CYAN}📦 FastAPI アプリケーションをデプロイ中...${NC}"
    
    sam build -t deploy/lambda-api.yml
    sam deploy \
        --template-file deploy/lambda-api.yml \
        --stack-name "satei-api-${env}" \
        --capabilities CAPABILITY_IAM \
        --resolve-s3 \
        --parameter-overrides "Environment=${env}" \
        --no-confirm-changeset
    
    # API Gateway URLを取得
    API_GATEWAY_URL=$(aws cloudformation describe-stacks \
        --stack-name "satei-api-${env}" \
        --query 'Stacks[0].Outputs[?OutputKey==`ValuationApi`].OutputValue' \
        --output text 2>/dev/null || echo "")
    
    if [[ -n "$API_GATEWAY_URL" ]]; then
        echo -e "${GREEN}✅ API デプロイ完了${NC}"
        echo -e "${CYAN}📍 API URL: ${API_GATEWAY_URL}${NC}"
        export API_GATEWAY_URL
    else
        echo -e "${YELLOW}⚠️  API URLの取得に失敗しました${NC}"
    fi
}

update_django_with_api_url() {
    local env=$1
    if [[ -n "$API_GATEWAY_URL" ]]; then
        echo -e "${CYAN}🔄 Django設定をAPI URLで更新中...${NC}"
        sam deploy \
            --template-file .aws-sam/build/template.yaml \
            --stack-name "satei-django-${env}" \
            --capabilities CAPABILITY_IAM \
            --resolve-s3 \
            --parameter-overrides "Environment=${env}" "ValuationApiUrl=${API_GATEWAY_URL}" \
            --no-confirm-changeset
        echo -e "${GREEN}✅ Django設定更新完了${NC}"
    fi
}

# 確認
echo -e "${YELLOW}デプロイを開始します...${NC}"
if [[ "$APP_TYPE" == "both" ]]; then
    echo "  🎯 Django + FastAPI を同時デプロイ"
elif [[ "$APP_TYPE" == "django" ]]; then
    echo "  🎯 Django のみデプロイ"
elif [[ "$APP_TYPE" == "api" ]]; then
    echo "  🎯 FastAPI のみデプロイ"
fi
echo "  🌍 環境: $ENVIRONMENT"
echo ""

read -p "デプロイを実行しますか？ (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}デプロイをキャンセルしました${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}🚀 デプロイを開始します...${NC}"
echo ""

# デプロイ実行
if [[ "$APP_TYPE" == "both" ]]; then
    # 同時デプロイ: API → Django の順序で実行
    echo -e "${PURPLE}📋 両方のアプリケーションを順番にデプロイします${NC}"
    echo ""
    
    # 1. APIをデプロイ
    deploy_api "$ENVIRONMENT"
    echo ""
    
    # 2. DjangoをAPIのURLを使ってデプロイ
    deploy_django "$ENVIRONMENT"
    echo ""
    
    # 3. API URLが取得できた場合、Djangoの設定を更新
    if [[ -n "$API_GATEWAY_URL" ]]; then
        update_django_with_api_url "$ENVIRONMENT"
    fi
    
elif [[ "$APP_TYPE" == "api" ]]; then
    deploy_api "$ENVIRONMENT"
elif [[ "$APP_TYPE" == "django" ]]; then
    deploy_django "$ENVIRONMENT"
fi

# デプロイ結果の表示
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}🎉 デプロイ完了！${NC}"
echo -e "${BLUE}============================================================${NC}"

echo ""
echo -e "${YELLOW}📊 デプロイ結果:${NC}"

if [[ "$APP_TYPE" == "django" || "$APP_TYPE" == "both" ]]; then
    echo ""
    echo -e "${CYAN}🌐 Django アプリケーション:${NC}"
    aws cloudformation describe-stacks \
        --stack-name "satei-django-${ENVIRONMENT}" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table 2>/dev/null || echo "Django スタック情報の取得に失敗"
fi

if [[ "$APP_TYPE" == "api" || "$APP_TYPE" == "both" ]]; then
    echo ""
    echo -e "${CYAN}🔧 FastAPI アプリケーション:${NC}"
    aws cloudformation describe-stacks \
        --stack-name "satei-api-${ENVIRONMENT}" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table 2>/dev/null || echo "API スタック情報の取得に失敗"
fi

# 次のステップ
echo ""
echo -e "${BLUE}🎯 次のステップ:${NC}"

if [[ "$APP_TYPE" == "both" ]]; then
    echo "1. 🌐 Django URL でフロントエンドの動作確認"
    echo "2. 🔧 API URL でバックエンドの動作確認"
    echo "3. 🔗 フロントエンドからAPIへの連携テスト"
    echo "4. 📊 /health エンドポイントでヘルスチェック"
    echo "5. 🏠 実際の不動産査定機能のテスト"
elif [[ "$APP_TYPE" == "django" ]]; then
    echo "1. 🌐 Django URL でアプリケーションの動作確認"
    echo "2. 🔗 API連携の設定確認"
    echo "3. 🎨 フロントエンド機能のテスト"
elif [[ "$APP_TYPE" == "api" ]]; then
    echo "1. 🔧 API URL でエンドポイントの動作確認"
    echo "2. 📊 /health エンドポイントでヘルスチェック"
    echo "3. 🏠 /api/valuation エンドポイントでMLモデルテスト"
fi

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}✨ 不動産査定アプリのLambdaデプロイが完了しました！${NC}"
echo -e "${BLUE}============================================================${NC}"