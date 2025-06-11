import os
import sys

# Djangoアプリケーションのパスを追加
sys.path.append('/opt/python')
sys.path.append('/opt/python/lib/python3.11/site-packages')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'satei_project.settings')

import django
from django.core.wsgi import get_wsgi_application
from mangum import Mangum

django.setup()

# DjangoアプリケーションのWSGI設定
django_app = get_wsgi_application()

# Lambda用ハンドラー
def lambda_handler(event, context):
    """
    Lambda関数のエントリーポイント
    """
    # Lambda Web Adapterを使用してDjangoアプリケーションを実行
    handler = Mangum(django_app, lifespan="off")
    return handler(event, context)