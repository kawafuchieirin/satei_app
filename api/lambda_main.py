from mangum import Mangum
from main import app

# Lambda用のハンドラー
handler = Mangum(app, lifespan="off")