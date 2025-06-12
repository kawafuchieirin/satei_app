#!/bin/bash

# SAMデプロイメント用スクリプト
# Lambda関数をデプロイするための設定

set -e

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE} 不動産査定アプリ SAMデプロイメント${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

# デフォルト設定
STACK_NAME=""
TEMPLATE_FILE=""
REGION="ap-northeast-1"
ENVIRONMENT="dev"
S3_BUCKET=""
GUIDED_MODE=false

# 使用方法を表示
usage() {
    echo "使用方法: $0 [OPTIONS]"
    echo ""
    echo "オプション:"
    echo "  -s, --stack-name NAME       スタック名（必須）"
    echo "  -t, --template FILE         テンプレートファイル（必須）"
    echo "  -r, --region REGION         AWSリージョン（デフォルト: ap-northeast-1）"
    echo "  -e, --environment ENV       環境（dev/prod）（デフォルト: dev）"
    echo "  -b, --s3-bucket BUCKET      S3バケット名（省略時は自動作成）"
    echo "  -g, --guided                ガイドモードでデプロイ"
    echo "  -h, --help                  ヘルプ表示"
    echo ""
    echo "例:"
    echo "  # Django Lambda（ガイドモード - 推奨）"
    echo "  $0 -s satei-django -t deploy/lambda-django.yml --guided"
    echo ""
    echo "  # FastAPI Lambda（S3バケット指定）"
    echo "  $0 -s satei-api -t deploy/lambda-api.yml -b my-sam-bucket"
    echo ""
    echo "  # 本番環境デプロイ"
    echo "  $0 -s satei-django -t deploy/lambda-django.yml -e prod --resolve-s3"
}

# パラメータ解析
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--stack-name)
            STACK_NAME="$2"
            shift 2
            ;;
        -t|--template)
            TEMPLATE_FILE="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -b|--s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        -g|--guided)
            GUIDED_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}不明なオプション: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# 必須パラメータのチェック
if [[ -z "$STACK_NAME" || -z "$TEMPLATE_FILE" ]]; then
    echo -e "${RED}エラー: スタック名とテンプレートファイルは必須です${NC}"
    usage
    exit 1
fi

# テンプレートファイルの存在確認
if [[ ! -f "$TEMPLATE_FILE" ]]; then
    echo -e "${RED}エラー: テンプレートファイルが見つかりません: $TEMPLATE_FILE${NC}"
    exit 1
fi

print_header

echo -e "${YELLOW}デプロイ設定:${NC}"
echo "  スタック名: $STACK_NAME"
echo "  テンプレート: $TEMPLATE_FILE"
echo "  リージョン: $REGION"
echo "  環境: $ENVIRONMENT"
echo ""

# SAMデプロイコマンドの構築
SAM_CMD="sam deploy"
SAM_CMD="$SAM_CMD --template-file $TEMPLATE_FILE"
SAM_CMD="$SAM_CMD --stack-name $STACK_NAME-$ENVIRONMENT"
SAM_CMD="$SAM_CMD --region $REGION"
SAM_CMD="$SAM_CMD --capabilities CAPABILITY_IAM"
SAM_CMD="$SAM_CMD --parameter-overrides Environment=$ENVIRONMENT"

# S3バケットの設定
if [[ "$GUIDED_MODE" = true ]]; then
    # ガイドモード
    echo -e "${GREEN}ガイドモードでデプロイを実行します${NC}"
    SAM_CMD="$SAM_CMD --guided"
elif [[ -n "$S3_BUCKET" ]]; then
    # 明示的なS3バケット指定
    echo -e "${GREEN}S3バケット使用: $S3_BUCKET${NC}"
    SAM_CMD="$SAM_CMD --s3-bucket $S3_BUCKET"
else
    # 自動でS3バケットを作成/使用
    echo -e "${GREEN}管理されたS3バケットを自動的に作成/使用します${NC}"
    SAM_CMD="$SAM_CMD --resolve-s3"
fi

# API URLの設定（lambda-django.ymlの場合）
if [[ "$TEMPLATE_FILE" == *"lambda-django.yml"* ]]; then
    # APIのURLを取得する必要がある場合
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        VALUATION_API_URL="https://your-prod-api-url.com"
    else
        VALUATION_API_URL="https://your-dev-api-url.com"
    fi
    SAM_CMD="$SAM_CMD ValuationApiUrl=$VALUATION_API_URL"
fi

# 確認
echo ""
echo -e "${YELLOW}実行コマンド:${NC}"
echo "$SAM_CMD"
echo ""

read -p "デプロイを実行しますか？ (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}デプロイをキャンセルしました${NC}"
    exit 0
fi

# デプロイ実行
echo ""
echo -e "${GREEN}デプロイを開始します...${NC}"

if eval $SAM_CMD; then
    echo ""
    echo -e "${GREEN}✅ デプロイが正常に完了しました！${NC}"
    
    # スタック出力の表示
    echo ""
    echo -e "${YELLOW}スタック出力:${NC}"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME-$ENVIRONMENT" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table
else
    echo ""
    echo -e "${RED}❌ デプロイに失敗しました${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}デプロイ完了！${NC}"
echo -e "${BLUE}============================================================${NC}"