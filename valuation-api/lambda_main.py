import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mangum import Mangum
from main import app

# Lambda用のハンドラー
handler = Mangum(app, lifespan="off")

# Lambda関数のエントリーポイント
def lambda_handler(event, context):
    return handler(event, context)