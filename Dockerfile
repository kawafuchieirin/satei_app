FROM python:3.11-slim

WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Djangoアプリケーションの依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 静的ファイルを収集
RUN python manage.py collectstatic --noinput || true

# ポート8080でアプリケーションを実行
EXPOSE 8080

# Lambda Web Adapterを使用してDjangoアプリケーションを実行
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "satei_project.wsgi:application"]