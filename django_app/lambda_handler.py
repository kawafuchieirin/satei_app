import os
import sys

# プロジェクトルートをパスに追加 
sys.path.insert(0, '/var/task')

# Django設定
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satei_project.settings')

import django
from django.core.asgi import get_asgi_application
from mangum import Mangum

# Djangoの初期化
django.setup()

# ASGIアプリケーション（Mangum用）
django_app = get_asgi_application()

# Mangumでラップ
handler = Mangum(django_app, lifespan="off")

def lambda_handler(event, context):
    """
    Lambda関数のエントリーポイント
    """
    # デバッグ情報をログに出力
    print(f"Event path: {event.get('path', 'No path')}")
    print(f"Event method: {event.get('httpMethod', 'No method')}")
    print(f"Lambda function name: {os.environ.get('AWS_LAMBDA_FUNCTION_NAME', 'Not set')}")
    print(f"Django settings module: {os.environ.get('DJANGO_SETTINGS_MODULE', 'Not set')}")
    
    # Mangumハンドラーを実行
    return handler(event, context)