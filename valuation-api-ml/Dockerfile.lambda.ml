FROM public.ecr.aws/lambda/python:3.9

# ML依存関係のインストール
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# アプリケーションコードのコピー
COPY . ${LAMBDA_TASK_ROOT}

# モデルファイルディレクトリの作成
RUN mkdir -p ${LAMBDA_TASK_ROOT}/models

# モデルクリエーションディレクトリは不要（Lambdaでは軽量モデルのみ使用）

# ハンドラーの設定
CMD ["lambda_main.handler"]
