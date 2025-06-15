FROM public.ecr.aws/lambda/python:3.11

# ML依存関係のインストール
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# アプリケーションコードのコピー
COPY . ${LAMBDA_TASK_ROOT}

# モデルファイルをコピー
COPY ../model-creation/models/*.joblib ${LAMBDA_TASK_ROOT}/models/

# ハンドラーの設定
CMD ["lambda_main.handler"]
