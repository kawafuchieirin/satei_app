#!/bin/bash

# 不動産査定アプリ 統合デプロイスクリプト
# 使用方法: ./deploy_unified.sh [DEPLOY_TYPE] [APP_TYPE] [ENVIRONMENT]
# DEPLOY_TYPE: aws | local
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
    echo -e "${PURPLE} 🚀 不動産査定アプリ 統合デプロイスクリプト${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

usage() {
    echo "使用方法: $0 [DEPLOY_TYPE] [APP_TYPE] [ENVIRONMENT]"
    echo ""
    echo "DEPLOY_TYPE:"
    echo "  aws       - AWS Lambda + API Gateway デプロイ (ZIP)"
    echo "  ecr       - AWS Lambda Container (ECR) デプロイ"
    echo "  local     - ローカル Docker Compose デプロイ"
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
    echo "  $0 ecr api dev        # ECR Container で API 開発環境デプロイ (推奨)"
    echo "  $0 aws both dev       # AWS Lambda に Django + API 開発環境デプロイ"
    echo "  $0 local both         # ローカル Docker で Django + API 起動"
    echo "  $0 ecr api prod       # ECR Container で API 本番環境デプロイ"
    echo ""
    echo "🎯 推奨:"
    echo "  - 開発時: 'local' でローカル環境での動作確認"
    echo "  - 本番時: 'ecr' でコンテナデプロイ（依存関係エラー解決）"
}

# パラメータの設定
DEPLOY_TYPE=${1:-"local"}
APP_TYPE=${2:-"both"}
ENVIRONMENT=${3:-"dev"}

# パラメータ検証
if [[ "$DEPLOY_TYPE" != "aws" && "$DEPLOY_TYPE" != "ecr" && "$DEPLOY_TYPE" != "local" ]]; then
    echo -e "${RED}エラー: DEPLOY_TYPEは 'aws', 'ecr', または 'local' である必要があります${NC}"
    usage
    exit 1
fi

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
echo "  デプロイタイプ: $DEPLOY_TYPE"
echo "  アプリタイプ: $APP_TYPE"
echo "  環境: $ENVIRONMENT"
echo ""

# 前提条件チェック
check_prerequisites() {
    echo -e "${YELLOW}前提条件チェック中...${NC}"
    
    if [[ "$DEPLOY_TYPE" == "aws" || "$DEPLOY_TYPE" == "ecr" ]]; then
        # AWS デプロイの前提条件
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
        
        # ECRデプロイの場合はDocker必須
        if [[ "$DEPLOY_TYPE" == "ecr" ]]; then
            if ! command -v docker &> /dev/null; then
                echo -e "${RED}❌ Dockerがインストールされていません${NC}"
                echo "インストール方法: https://docs.docker.com/get-docker/"
                exit 1
            fi
            
            if ! docker info &> /dev/null; then
                echo -e "${RED}❌ Dockerデーモンが起動していません${NC}"
                echo "Docker Desktop を起動してください"
                exit 1
            fi
        fi
        
        # AWS認証確認
        if ! aws sts get-caller-identity &> /dev/null; then
            echo -e "${RED}❌ AWS認証が設定されていません${NC}"
            echo "aws configure を実行してください"
            exit 1
        fi
        
    elif [[ "$DEPLOY_TYPE" == "local" ]]; then
        # ローカル デプロイの前提条件
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}❌ Dockerがインストールされていません${NC}"
            echo "インストール方法: https://docs.docker.com/get-docker/"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
            echo -e "${RED}❌ Docker Composeがインストールされていません${NC}"
            echo "インストール方法: https://docs.docker.com/compose/install/"
            exit 1
        fi
        
        # Dockerデーモン確認
        if ! docker info &> /dev/null; then
            echo -e "${RED}❌ Dockerデーモンが起動していません${NC}"
            echo "Docker Desktop を起動してください"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}✅ 前提条件OK${NC}"
    echo ""
}

# AWS デプロイ関数
deploy_aws_django() {
    local env=$1
    echo -e "${CYAN}📦 Django アプリケーションを AWS Lambda にデプロイ中...${NC}"
    
    # プロジェクトルートに移動
    cd ..
    
    # API Gateway URL を設定（APIを先にデプロイする場合）
    API_URL="http://localhost:8000"
    if [[ "$APP_TYPE" == "both" ]] || [[ -n "$API_GATEWAY_URL" ]]; then
        API_URL="${API_GATEWAY_URL:-https://api-placeholder.execute-api.ap-northeast-1.amazonaws.com}"
    fi
    
    sam build -t deploy/lambda-django.yml
    sam deploy \
        --template-file deploy/lambda-django.yml \
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
        echo -e "${GREEN}✅ Django AWS デプロイ完了${NC}"
        echo -e "${CYAN}📍 Django URL: ${DJANGO_URL}${NC}"
    else
        echo -e "${YELLOW}⚠️  Django URLの取得に失敗しました${NC}"
    fi
    
    # deployディレクトリに戻る
    cd deploy
}

deploy_aws_api() {
    local env=$1
    echo -e "${CYAN}📦 FastAPI アプリケーションを AWS Lambda にデプロイ中...${NC}"
    
    # プロジェクトルートに移動
    cd ..
    
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
        echo -e "${GREEN}✅ FastAPI AWS デプロイ完了${NC}"
        echo -e "${CYAN}📍 API URL: ${API_GATEWAY_URL}${NC}"
        export API_GATEWAY_URL
    else
        echo -e "${YELLOW}⚠️  API URLの取得に失敗しました${NC}"
    fi
    
    # deployディレクトリに戻る
    cd deploy
}

deploy_ecr_api() {
    local env=$1
    echo -e "${CYAN}📦 FastAPI アプリケーションを ECR Container でデプロイ中...${NC}"
    
    # ECRデプロイスクリプトを実行
    ./ecr-deploy.sh $env
    
    # API Gateway URLを取得
    API_GATEWAY_URL=$(aws cloudformation describe-stacks \
        --stack-name "satei-api-container-${env}" \
        --query 'Stacks[0].Outputs[?OutputKey==`ValuationApi`].OutputValue' \
        --output text 2>/dev/null || echo "")
    
    if [[ -n "$API_GATEWAY_URL" ]]; then
        echo -e "${GREEN}✅ FastAPI ECR デプロイ完了${NC}"
        echo -e "${CYAN}📍 API URL: ${API_GATEWAY_URL}${NC}"
        export API_GATEWAY_URL
    else
        echo -e "${YELLOW}⚠️  API URLの取得に失敗しました${NC}"
    fi
}

# ローカル デプロイ関数
deploy_local() {
    local app_type=$1
    echo -e "${CYAN}📦 ローカル Docker Compose で起動中...${NC}"
    
    # Docker Compose を使用してサービスを起動
    if [[ "$app_type" == "both" ]]; then
        echo -e "${PURPLE}🐳 Django + FastAPI の両方をローカルで起動${NC}"
        docker-compose up --build -d
        
        echo -e "${GREEN}✅ ローカルデプロイ完了${NC}"
        echo -e "${CYAN}📍 Django URL: http://localhost:8080${NC}"
        echo -e "${CYAN}📍 FastAPI URL: http://localhost:8000${NC}"
        echo -e "${CYAN}📍 FastAPI Docs: http://localhost:8000/docs${NC}"
        
    elif [[ "$app_type" == "django" ]]; then
        echo -e "${PURPLE}🐳 Django のみローカルで起動${NC}"
        docker-compose up --build -d django
        
        echo -e "${GREEN}✅ Django ローカルデプロイ完了${NC}"
        echo -e "${CYAN}📍 Django URL: http://localhost:8080${NC}"
        
    elif [[ "$app_type" == "api" ]]; then
        echo -e "${PURPLE}🐳 FastAPI のみローカルで起動${NC}"
        docker-compose up --build -d api
        
        echo -e "${GREEN}✅ FastAPI ローカルデプロイ完了${NC}"
        echo -e "${CYAN}📍 FastAPI URL: http://localhost:8000${NC}"
        echo -e "${CYAN}📍 FastAPI Docs: http://localhost:8000/docs${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}📋 ローカル環境コマンド:${NC}"
    echo "  ログ確認: docker-compose logs -f"
    echo "  停止: docker-compose down"
    echo "  リスタート: docker-compose restart"
}

update_django_with_api_url() {
    local env=$1
    if [[ -n "$API_GATEWAY_URL" ]]; then
        echo -e "${CYAN}🔄 Django設定をAPI URLで更新中...${NC}"
        cd ..
        sam deploy \
            --template-file deploy/lambda-django.yml \
            --stack-name "satei-django-${env}" \
            --capabilities CAPABILITY_IAM \
            --resolve-s3 \
            --parameter-overrides "Environment=${env}" "ValuationApiUrl=${API_GATEWAY_URL}" \
            --no-confirm-changeset
        echo -e "${GREEN}✅ Django設定更新完了${NC}"
        cd deploy
    fi
}

# 前提条件チェック実行
check_prerequisites

# 確認
echo -e "${YELLOW}デプロイを開始します...${NC}"
echo "  🚀 デプロイタイプ: $DEPLOY_TYPE"
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
if [[ "$DEPLOY_TYPE" == "aws" ]]; then
    # AWS デプロイ
    if [[ "$APP_TYPE" == "both" ]]; then
        # 同時デプロイ: API → Django の順序で実行
        echo -e "${PURPLE}📋 AWS Lambda に両方のアプリケーションを順番にデプロイします${NC}"
        echo ""
        
        # 1. APIをデプロイ
        deploy_aws_api "$ENVIRONMENT"
        echo ""
        
        # 2. DjangoをAPIのURLを使ってデプロイ
        deploy_aws_django "$ENVIRONMENT"
        echo ""
        
        # 3. API URLが取得できた場合、Djangoの設定を更新
        if [[ -n "$API_GATEWAY_URL" ]]; then
            update_django_with_api_url "$ENVIRONMENT"
        fi
        
    elif [[ "$APP_TYPE" == "api" ]]; then
        deploy_aws_api "$ENVIRONMENT"
    elif [[ "$APP_TYPE" == "django" ]]; then
        deploy_aws_django "$ENVIRONMENT"
    fi
    
elif [[ "$DEPLOY_TYPE" == "ecr" ]]; then
    # ECR デプロイ
    if [[ "$APP_TYPE" == "api" ]]; then
        deploy_ecr_api "$ENVIRONMENT"
    else
        echo -e "${YELLOW}⚠️  ECR デプロイは現在APIのみサポートしています${NC}"
        echo -e "${YELLOW}    Django の場合は 'aws' デプロイを使用してください${NC}"
        exit 1
    fi
    
elif [[ "$DEPLOY_TYPE" == "local" ]]; then
    # ローカル デプロイ
    deploy_local "$APP_TYPE"
fi

# デプロイ結果の表示
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}🎉 デプロイ完了！${NC}"
echo -e "${BLUE}============================================================${NC}"

if [[ "$DEPLOY_TYPE" == "aws" || "$DEPLOY_TYPE" == "ecr" ]]; then
    echo ""
    echo -e "${YELLOW}📊 AWS デプロイ結果:${NC}"
    
    if [[ "$APP_TYPE" == "django" || "$APP_TYPE" == "both" ]]; then
        echo ""
        echo -e "${CYAN}🌐 Django アプリケーション:${NC}"
        cd ..
        aws cloudformation describe-stacks \
            --stack-name "satei-django-${ENVIRONMENT}" \
            --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
            --output table 2>/dev/null || echo "Django スタック情報の取得に失敗"
        cd deploy
    fi
    
    if [[ "$APP_TYPE" == "api" || "$APP_TYPE" == "both" ]]; then
        echo ""
        echo -e "${CYAN}🔧 FastAPI アプリケーション:${NC}"
        cd ..
        if [[ "$DEPLOY_TYPE" == "ecr" ]]; then
            STACK_NAME="satei-api-container-${ENVIRONMENT}"
        else
            STACK_NAME="satei-api-${ENVIRONMENT}"
        fi
        aws cloudformation describe-stacks \
            --stack-name "$STACK_NAME" \
            --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
            --output table 2>/dev/null || echo "API スタック情報の取得に失敗"
        cd deploy
    fi
fi

# 次のステップ
echo ""
echo -e "${BLUE}🎯 次のステップ:${NC}"

if [[ "$DEPLOY_TYPE" == "aws" || "$DEPLOY_TYPE" == "ecr" ]]; then
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
elif [[ "$DEPLOY_TYPE" == "local" ]]; then
    echo "1. 🌐 http://localhost:8080 でフロントエンドの動作確認"
    echo "2. 🔧 http://localhost:8000 でバックエンドの動作確認"
    echo "3. 📚 http://localhost:8000/docs で API ドキュメント確認"
    echo "4. 🔍 docker-compose logs -f でログ確認"
    echo "5. 🏠 実際の不動産査定機能のテスト"
fi

echo ""
echo -e "${BLUE}============================================================${NC}"
if [[ "$DEPLOY_TYPE" == "aws" || "$DEPLOY_TYPE" == "ecr" ]]; then
    if [[ "$DEPLOY_TYPE" == "ecr" ]]; then
        echo -e "${GREEN}✨ 不動産査定アプリの ECR Container デプロイが完了しました！${NC}"
    else
        echo -e "${GREEN}✨ 不動産査定アプリの AWS デプロイが完了しました！${NC}"
    fi
else
    echo -e "${GREEN}✨ 不動産査定アプリのローカル環境が起動しました！${NC}"
fi
echo -e "${BLUE}============================================================${NC}"