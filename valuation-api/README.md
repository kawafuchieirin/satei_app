# 査定API (FastAPI Backend)

このディレクトリには不動産査定のML APIサービスが含まれています。

## 主要ファイル
- **main.py**: FastAPI アプリケーションのエントリーポイント
- **lambda_main.py**: AWS Lambda 用ハンドラー
- **models/**: ML モデル関連クラス
- **requirements.txt**: Python 依存関係

## API エンドポイント
- `/`: API ルート
- `/api/valuation`: 査定エンドポイント
- `/health`: ヘルスチェック
- `/api/model/evaluate`: モデル評価

## ローカル実行
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
