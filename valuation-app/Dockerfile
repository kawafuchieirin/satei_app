FROM python:3.11-slim

WORKDIR /app

# システムパッケージの更新
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションのコピー
COPY . .

# データベースの初期化
RUN python manage.py migrate

# ポート公開
EXPOSE 8080

# アプリケーション起動
CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]