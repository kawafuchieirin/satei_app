#!/bin/bash

# Lambda Layer作成スクリプト
# 依存関係をLayerとして分離

set -e

echo "🔧 Lambda Layer for Python dependencies を作成中..."

# 作業ディレクトリ作成
mkdir -p lambda-layer/python

# requirements.txtをコピー
cp ../api/requirements.txt lambda-layer/

# Dockerコンテナで依存関係をインストール
docker run --rm -v $(pwd)/lambda-layer:/workspace -w /workspace \
  public.ecr.aws/sam/build-python3.11:latest \
  pip install -r requirements.txt -t python/

# Layer用ZIPファイル作成
cd lambda-layer
zip -r ../satei-dependencies-layer.zip python/
cd ..

echo "✅ Layer ZIP作成完了: satei-dependencies-layer.zip"

# AWS CLIでLayerをデプロイ
echo "🚀 Lambda Layerをデプロイ中..."

LAYER_ARN=$(aws lambda publish-layer-version \
  --layer-name satei-python-dependencies \
  --zip-file fileb://satei-dependencies-layer.zip \
  --compatible-runtimes python3.11 \
  --description "Python dependencies for Satei app" \
  --query 'LayerVersionArn' \
  --output text)

echo "✅ Lambda Layer作成完了!"
echo "📍 Layer ARN: $LAYER_ARN"

# 結果をファイルに保存
echo $LAYER_ARN > layer-arn.txt

echo ""
echo "🎯 次のステップ:"
echo "1. SAMテンプレートでこのLayerを使用"
echo "2. requirements.txtから依存関係を削除"
echo "3. Lambdaをデプロイ"

# クリーンアップ
rm -rf lambda-layer satei-dependencies-layer.zip