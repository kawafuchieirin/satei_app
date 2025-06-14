from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List, Dict
import os
import sys

# 共通モジュールの追加
sys.path.append('/app')
# 緊急対応：簡易版MLPredictorを強制使用
from valuation_core.ml_predictor_simple import MLPredictorSimple as MLPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 共通MLPredicatorを使用
try:
    valuation_model = MLPredictor(model_path='/app/models')
    ML_AVAILABLE = True
    logger.info("Using shared MLPredictor with ML capabilities.")
except Exception as e:
    logger.error(f"MLPredictor initialization failed: {e}")
    valuation_model = None
    ML_AVAILABLE = False

app = FastAPI(
    title="Real Estate Valuation API (Docker Compose)",
    description="API for real estate property valuation using shared ML models",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PropertyData(BaseModel):
    prefecture: str
    city: str
    district: Optional[str] = ""
    land_area: float
    building_area: float
    building_age: int

class ValuationResponse(BaseModel):
    estimated_price: float
    confidence: Optional[float] = None
    price_range: Optional[Dict[str, float]] = None
    factors: Optional[List[str]] = None

@app.get("/")
async def root():
    return {"message": "Real Estate Valuation API (Docker Compose)", "status": "running"}

@app.get("/health")
async def health_check():
    """
    ヘルスチェック - MLモデルの状態も含む
    """
    ml_status = "available" if (valuation_model and valuation_model.is_model_available()) else "unavailable"
    return {
        "status": "healthy",
        "ml_model": ml_status,
        "service_type": "Shared MLPredictor",
        "version": "2.0.0"
    }

@app.post("/api/valuation", response_model=ValuationResponse)
async def predict_valuation(property_data: PropertyData):
    """
    共通MLPredictor専用査定エンドポイント
    MLモデルが利用できない場合は503エラーを返す
    """
    try:
        logger.info(f"Received valuation request: {property_data}")
        
        # MLモデル利用可能性の事前チェック
        if not valuation_model or not valuation_model.is_model_available():
            logger.error("ML model not available for prediction")
            raise HTTPException(status_code=503, detail="査定できませんでした。MLモデルが利用できません。")
        
        # 共通MLPredictorによる予測実行
        result = valuation_model.predict(
            prefecture=property_data.prefecture,
            city=property_data.city,
            land_area=property_data.land_area,
            building_area=property_data.building_area,
            building_age=property_data.building_age,
            district=property_data.district or ""
        )
        
        logger.info(f"ML valuation result: {result}")
        return result
        
    except HTTPException:
        # HTTPExceptionは再発生（503/422/500など）
        raise
    except Exception as e:
        logger.error(f"Unexpected error during valuation: {str(e)}")
        raise HTTPException(status_code=500, detail="システムエラーが発生しました。しばらくしてから再度お試しください。")

@app.get("/api/model/info")
async def get_model_info():
    """
    現在のモデル情報を取得
    """
    try:
        if not valuation_model:
            return {"error": "Model not initialized"}
            
        info = {
            "model_loaded": valuation_model.is_trained,
            "model_available": valuation_model.is_model_available(),
            "model_type": "SharedMLPredictor",
            "features": valuation_model.feature_columns,
            "version": "2.0.0"
        }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)