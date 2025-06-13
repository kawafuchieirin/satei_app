FROM python:3.11-slim

WORKDIR /app

# システムの依存関係をインストール
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# APIの依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# ポート8000でAPIを実行
EXPOSE 8000

# FastAPIアプリケーションを実行
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]