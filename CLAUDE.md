# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a microservices-based real estate valuation application for Tokyo's 23 wards, with Django frontend and FastAPI backend deployed on AWS Lambda.

### Docker Compose 開発環境
ローカル開発では、本番と同等のMLロジックを動作させるためのDocker Compose環境を構築済み：

```
┌─────────────────┐    HTTP     ┌─────────────────┐
│  Django App     │ ──────────→ │  FastAPI App    │
│  (Port 8000)    │             │  (Port 8001)    │
│  - UI管理       │             │  - ML査定API    │
│  - フォーム     │             │  - 共通モジュール │
└─────────────────┘             └─────────────────┘
         │                               │
         └───────── valuation_core ──────┘
              (共通MLPredictor - 統一ロジック)
```

#### 共通モジュール構成
- **valuation_core/ml_predictor.py**: Django・FastAPI共通のMLロジック
- **統一性確保**: 本番とローカルで同一の予測アルゴリズム
- **開発効率**: コードの重複排除、バグ修正の一元化

### Service Communication Flow
```
[User] → [Django Lambda] → HTTP POST → [FastAPI Lambda] → [Random Forest/XGBoost Only]
                ↓                            ↓                        ↓
        [Form Validation]            [ML Model Required]      [High Precision: R²=0.8474]
                ↓                            ↓                        ↓
        [Result Display] ← JSON ← [Price Prediction] ← [63,217件データ訓練済み]
                ↓
        [503 Error if ML Unavailable]
```

### Production URLs
- **Django**: https://imi1rg1eyc.execute-api.ap-northeast-1.amazonaws.com/Prod/
- **FastAPI (ML-Enabled)**: https://25cfdqih7a.execute-api.ap-northeast-1.amazonaws.com/Prod/ ✅ (ECRコンテナ版、ML専用)

### Current Model State
- **Production**: ECRコンテナ版ML API（Random Forest/XGBoostモデル専用）
- **ML API**: 118.8MB訓練済みモデル搭載、3GB RAM、5分タイムアウト
- **Model Performance**: XGBoost R² = 0.8474, Random Forest R² = 0.8278
- **Data Source**: 63,217件の東京23区実取引データ（2022-2024年）
- **Error Handling**: MLモデル利用不可時は適切なHTTPエラー（503/500）とエラーメッセージを返却
- **No Rule-based Fallback**: ルールベース査定は使用せず、MLモデルのみで動作

## Common Development Commands

### Local Development

#### 🐳 Docker Compose 環境 (推奨)
```bash
# 統合開発環境（Django + FastAPI）
docker-compose up --build -d

# アクセス
# Django UI: http://localhost:8000
# FastAPI API: http://localhost:8001/docs

# ログ確認
docker-compose logs -f

# 停止
docker-compose down
```

#### 🔧 個別起動 (レガシー)
```bash
# Run Django development server only
cd valuation-app
python manage.py runserver 0.0.0.0:8080

# Run FastAPI development server only  
cd valuation-api-ml
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Model Management

#### ML Model Training (XGBoost専用)
```bash
cd model-creation

# データの準備と統一ファイル作成
python tokyo23_data_fetcher.py              # MLIT APIから最新データ取得 → tokyo23_real_estate.csv

# XGBoostモデル訓練（唯一の訓練スクリプト）
python train_xgboost_model.py               # XGBoost + Random Forest + 線形回帰の比較訓練

# 訓練済みモデルをAPIに統合
python deploy_model.py                      # models/ → valuation-api-ml/
```

#### Model Performance Benchmarks
```bash
# 現在の性能指標（tokyo23_real_estate.csvで訓練）
# XGBoost:       R² = 0.8474, RMSE = 26,024,318円, MAPE = 20.90%
# Random Forest: R² = 0.8278, RMSE = 27,642,348円, MAPE = 21.01%
# Linear Reg:    R² = 0.4675, RMSE = 48,611,593円, MAPE = 61.14%
```

#### Model Testing & Validation
```bash
# APIエンドポイントテスト
python test_api_connection.py               # ローカル・本番API疎通確認

# 予測テスト例（Docker Compose環境）
curl -X POST http://localhost:8001/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"港区","district":"六本木","land_area":150,"building_area":120,"building_age":5}'

# 予測テスト例（本番環境）
curl -X POST https://25cfdqih7a.execute-api.ap-northeast-1.amazonaws.com/Prod/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"港区","district":"六本木","land_area":150,"building_area":120,"building_age":5}'
```

### Deployment Commands

#### Unified Deployment (推奨)
```bash
cd deployment
./deploy_all.sh both prod             # Django + 軽量API の同時デプロイ
./deploy_all.sh api prod              # 軽量API のみ
./deploy_all.sh django prod           # Django のみ
```

#### ML対応ECRデプロイ（本格運用向け）
```bash
cd deployment
./ecr-deploy-ml.sh api prod           # ML完全版APIをECRコンテナでデプロイ
                                       # - 3GB RAM, 5分タイムアウト
                                       # - 118.8MB訓練済みモデル搭載
                                       # - XGBoost, Random Forest対応
```

#### レガシー個別デプロイ
```bash
# 軽量版（現在の本番環境）
sam build -t lambda-api-light.yml
sam deploy --stack-name satei-api-light --resolve-s3

# Lambda Layer管理
./create-lambda-layer.sh              # 共通依存関係のレイヤー作成
```

#### デプロイ状況確認
```bash
# CloudFormationスタック確認
aws cloudformation describe-stacks --stack-name satei-api-light      # 軽量版
aws cloudformation describe-stacks --stack-name satei-api-ml-prod    # ML版

# Lambda関数一覧
aws lambda list-functions --query "Functions[?contains(FunctionName, 'satei')]"

# 各API疎通確認
curl https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/     # 軽量版
curl https://25cfdqih7a.execute-api.ap-northeast-1.amazonaws.com/Prod/     # ML版
```

### Testing
```bash
# Django application tests
cd valuation-app
python manage.py test valuation

# API endpoint tests (note: district parameter is optional)
curl -X POST https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"渋谷区","land_area":100,"building_area":80,"building_age":10}'

# Test local environment
curl http://localhost:8000/api/valuation -X POST -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"港区","land_area":150,"building_area":120,"building_age":5}'
```

## High-Level Architecture

### Service Dependencies and Deployment Strategy
The Django frontend depends on the FastAPI backend URL specified in `VALUATION_API_URL`. The `deploy_all.sh` script handles this complexity automatically:

**Deployment Order (Automatic):**
1. Deploy FastAPI first to get its URL
2. Deploy Django with the FastAPI URL injected as environment variable
3. Update Django configuration if FastAPI URL changes

**Manual Environment Variable Management:**
```bash
# Check current Lambda environment variables
aws lambda get-function-configuration --function-name satei-django-prod --query 'Environment.Variables'

# Update Django Lambda to point to correct API
aws lambda update-function-configuration --function-name satei-django-prod \
  --environment Variables='{VALUATION_API_URL=https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod,...}'
```

### Lambda Adaptation Pattern
Both services use Mangum for ASGI-to-Lambda adaptation:
```python
# Django (lambda_handler.py)
django_app = get_asgi_application()
handler = Mangum(django_app, lifespan="off")

# FastAPI (lambda_main.py)  
handler = Mangum(app, lifespan="off")
```

### Model Architecture and Deployment Strategy

**Current Production Environment (2-tier system):**

1. **軽量版API (現在の本番)**
   - URL: https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod/
   - Lambda ZIP (250MB制限)
   - **Random Forest/XGBoostモデルのみ使用**
   - MLモデル利用不可時は503エラー「査定できませんでした」
   - ルールベース査定は使用しない
   - Django → この軽量版APIを呼び出し

2. **ML完全版API (構築済み、調整中)**
   - URL: https://25cfdqih7a.execute-api.ap-northeast-1.amazonaws.com/Prod/
   - ECRコンテナ (サイズ制限なし)
   - 118.8MB訓練済みモデル搭載
   - 3GB RAM, 5分タイムアウト
   - XGBoost (R² = 0.8474) + Random Forest対応

**Model Files & Performance:**
```bash
# 統合済みモデルファイル
valuation-api-ml/
├── valuation_model.joblib      # 118.8MB (Random Forest/XGBoost)
├── label_encoders.joblib       # 0.04MB (カテゴリエンコーダー)
├── scaler.joblib              # 0.001MB (数値正規化)
└── feature_columns.joblib     # 0.0002MB (特徴量定義)

# データソース
model-creation/data/tokyo23_real_estate.csv  # 63,217件, 11.8MB
```

**Model Training Data:**
- **期間**: 2022-2024年（3年間）
- **地域**: 東京23区全域
- **件数**: 63,217件の実取引データ
- **ソース**: 国土交通省不動産取引価格情報API
- **特徴量**: 10次元（都道府県、市区町村、地区、土地面積、建物面積、築年数、建ぺい率、容積率、建物構造、用途）

### Lambda Configuration Adjustments
Django automatically detects Lambda environment and adjusts:
- Disables CSRF middleware (commented out in settings.py)
- Sets `FORCE_SCRIPT_NAME = '/Prod'` for API Gateway routing
- Configures CORS trusted origins

### ML Model Integration Points

**Data Pipeline:**
1. **Data Fetching**: `tokyo23_data_fetcher.py` → MLIT API → `tokyo23_real_estate.csv`
2. **Preprocessing**: `data_preprocessor.py` → Feature engineering (63,217 → 10 features)
3. **Training**: `train_xgboost_model.py` → Multi-model comparison → Best model selection
4. **Deployment**: `deploy_model.py` → Copy trained models to API directories

**Model Loading & Inference:**
```python
# valuation-api/models/lightweight_model.py
class LightweightValuationModel:
    def __init__(self):
        self._try_load_ml_model()  # Dynamic import for Lambda safety
        
    def predict(self, prefecture, city, land_area, building_area, building_age, district=""):
        # Random Forest/XGBoostモデルのみ使用
        if self.ml_available and self.ml_model is not None:
            return self._ml_predict(...)  # XGBoost/Random Forest
        else:
            # MLモデルが利用できない場合はエラー
            raise RuntimeError("査定できませんでした。MLモデルが利用できません。")
```

**Error Handling Strategy:**
- **ML Model Available**: Random Forest/XGBoostによる高精度予測
- **ML Model Unavailable**: HTTP 503 + "査定できませんでした。MLモデルが利用できません。"
- **Invalid Input**: HTTP 422 + バリデーションエラー詳細
- **System Error**: HTTP 500 + 汎用エラーメッセージ
- **No Rule-based Fallback**: ルールベース査定は実装せず、MLモデル専用

## Data Management

### tokyo23_real_estate.csv
```bash
# 統一データファイル（タイムスタンプなし、上書き更新）
model-creation/data/tokyo23_real_estate.csv

# データ更新（必要時のみ実行）
cd model-creation
python tokyo23_data_fetcher.py  # 最新データで上書き保存

# データ統計
# - 件数: 63,217件
# - 期間: 2022-2024年
# - サイズ: 11.8MB
# - カラム: 22項目（Prefecture, Municipality, TradePrice等）
```

### .gitignore Configuration
```bash
# 大容量データファイルはgit管理対象外
data/
data/raw/
data/processed/
*.csv
!sample_data.csv
models/trained/
*.joblib
```

## ML Model Lifecycle

### Complete Training & Deployment Workflow
```bash
# 1. 最新データ取得
cd model-creation
python tokyo23_data_fetcher.py

# 2. モデル訓練・比較
python train_xgboost_model.py       # XGBoost最高性能: R² = 0.8474

# 3. API統合
python deploy_model.py              # → valuation-api/ & valuation-api-ml/

# 4. 軽量版デプロイ (現在の本番)
cd ../deployment
./deploy_all.sh api prod

# 5. ML完全版デプロイ (将来の本番)
./ecr-deploy-ml.sh api prod
```

### Model Performance Benchmarks
| Model | R² Score | RMSE (円) | MAPE (%) | File Size |
|-------|----------|-----------|----------|-----------|
| XGBoost | 0.8474 | 26,024,318 | 20.90 | 118.8MB |
| Random Forest | 0.8278 | 27,642,348 | 21.01 | 118.8MB |
| Quick RF | 0.8191 | 28,331,550 | 22.26 | 25.9MB |
| Linear Regression | 0.4675 | 48,611,593 | 61.14 | 0.1MB |

## Critical Configuration Notes

### API Request/Response Format
```python
# Request (district is optional)
{
    "prefecture": "東京都",
    "city": "渋谷区",
    "land_area": 100,
    "building_area": 80,
    "building_age": 10
}

# Response (正常時 - Random Forest/XGBoost)
{
    "estimated_price": 24909.89,
    "confidence": 87.0,
    "price_range": {"min": 21173.41, "max": 28646.38},
    "factors": [
        "機械学習モデルによる高精度予測",
        "Random Forest/XGBoost アルゴリズム使用",
        "63,217件の実取引データで訓練",
        "築浅物件で、価格にプラス影響",
        "土地面積が広く、価格にプラス影響"
    ]
}

# Response (MLモデル利用不可時)
HTTP 503: {"detail": "査定できませんでした。MLモデルが利用できません。"}

# Response (入力エラー時)
HTTP 422: {"detail": "入力データが無効です: 土地面積は正の数値である必要があります"}

# Response (システムエラー時)
HTTP 500: {"detail": "システムエラーが発生しました。しばらくしてから再度お試しください"}
```

### Valuation Logic & Error Handling

**Random Forest/XGBoost MLモデル専用査定:**
```python
# MLモデルによる査定のみ
def _ml_predict(self, prefecture, city, land_area, building_area, building_age, district):
    # 1. 特徴量エンコーディング
    input_data = pd.DataFrame({
        'prefecture': [prefecture], 'city': [city], 'district': [district],
        'land_area': [land_area], 'building_area': [building_area], 
        'building_age': [building_age]
    })
    
    # 2. カテゴリカル変数の数値化
    for col in ['prefecture', 'city', 'district']:
        input_data[f'{col}_encoded'] = self.ml_encoders[col].transform(input_data[col])
    
    # 3. 派生特徴量の生成
    input_data['total_area'] = input_data['land_area'] + input_data['building_area']
    input_data['building_ratio'] = input_data['building_area'] / (input_data['land_area'] + 1)
    
    # 4. Random Forest/XGBoostによる予測
    predicted_price = self.ml_model.predict(X)[0]
    
    return predicted_price

# エラー処理: MLモデル利用不可時は例外発生
if not self.ml_available:
    raise RuntimeError("査定できませんでした。MLモデルが利用できません。")
```

**厳格なエラーハンドリング:**
- **MLモデル必須**: Random Forest/XGBoostモデルが利用できない場合は503エラー
- **入力バリデーション**: 必須フィールド、数値範囲、東京23区チェック
- **No Fallback**: ルールベース査定は実装せず、MLモデル専用動作
- **詳細ログ**: MLモデルの読み込み状況、予測エラーをCloudWatchに記録

### Package Size & Dependency Management

**Environment-Specific Requirements:**
```bash
# 軽量版Lambda (ZIP)
requirements-lambda.txt         # FastAPI, mangum only (4.5MB)

# ML完全版Lambda (ECR)
requirements-ml.txt            # + pandas, numpy, scikit-learn, xgboost (500MB)

# ローカル開発
requirements.txt              # Full development stack
```

**Model File Size Optimization:**
- **Full Model**: 118.8MB (production ECR)
- **Quick Model**: 25.9MB (development)
- **Feature Engineering**: 10 dimensions (optimized)
- **Lambda Memory**: 3GB (ML processing対応)
- **Timeout**: 5分 (model loading + inference)

### Environment-Specific Settings

**Lambda Environment Detection:**
```python
# Auto-detection for Lambda environment
if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
    # Lambda-specific configurations
    FORCE_SCRIPT_NAME = '/Prod'
    CSRF_MIDDLEWARE = False  # Disabled for API Gateway
    
# ML Model availability check
try:
    import joblib, pandas, numpy
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
```

**API URL Configuration:**
```bash
# Django環境変数 (CloudFormation経由で注入)
VALUATION_API_URL="https://tal7iqok0h.execute-api.ap-northeast-1.amazonaws.com/Prod"  # 現在
# VALUATION_API_URL="https://25cfdqih7a.execute-api.ap-northeast-1.amazonaws.com/Prod"  # ML版準備完了
```

**Critical Environment Variables:**
- `MLIT_API_KEY`: 国土交通省API認証キー
- `MODEL_PATH`: "/var/task/models" (Lambda) or "./models" (local)
- `ENABLE_ML`: "true" (ML version) or "false" (lightweight)
- `LOG_LEVEL`: "INFO" (production) or "DEBUG" (development)

## Docker Compose 統合開発環境

### ディレクトリ構成
```bash
satei_app/
├── docker-compose.yml          # マルチコンテナ環境定義
├── .env                        # 環境変数設定
├── valuation_core/             # 🔥 共通MLモジュール
│   ├── __init__.py
│   └── ml_predictor.py         # Django・FastAPI統一ロジック
├── django_app/                 # Django UI (Port 8000)
│   ├── Dockerfile
│   └── (Django application)
├── fastapi_app/                # FastAPI ML API (Port 8001)  
│   ├── Dockerfile
│   ├── main_shared.py          # 共通MLPredictor使用版
│   └── (FastAPI application)
└── model-creation/models/      # MLモデルファイル共有
    ├── valuation_model.joblib  # 4.1MB最適化Random Forest
    ├── label_encoders.joblib   # カテゴリエンコーダー
    └── scaler.joblib          # 特徴量スケーラー
```

### 開発環境の利点
1. **本番同等ロジック**: AWS LambdaとローカルでMLロジック統一
2. **開発効率向上**: Hot reload、独立スケーリング
3. **テスト容易性**: 完全なE2E環境での検証
4. **ゼロ設定Gap**: 本番とローカルの査定結果差分ゼロ

### 検証コマンド
```bash
# 環境起動
docker-compose up --build -d

# 査定API直接テスト
curl -X POST http://localhost:8001/api/valuation \
  -H "Content-Type: application/json" \
  -d '{"prefecture":"東京都","city":"渋谷区","land_area":100,"building_area":80,"building_age":10}'

# Django経由のE2Eテスト
curl http://localhost:8000/test-api/

# ヘルスチェック
curl http://localhost:8001/health
```