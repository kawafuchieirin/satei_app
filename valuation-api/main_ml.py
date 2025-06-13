from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# モデルの動的インポート
try:
    from models.valuation_model import ValuationModel
    ML_AVAILABLE = True
    logger.info("Using full ML model.")
except ImportError:
    from models.lightweight_model import LightweightValuationModel
    ML_AVAILABLE = False
    logger.info("Using lightweight model for Lambda environment.")

app = FastAPI(
    title="Real Estate Valuation API",
    description="API for real estate property valuation using ML models",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルの初期化
if ML_AVAILABLE:
    valuation_model = ValuationModel()
else:
    valuation_model = LightweightValuationModel()


class PropertyData(BaseModel):
    prefecture: str
    city: str
    district: str
    land_area: float
    building_area: float
    building_age: int


class ValuationResponse(BaseModel):
    estimated_price: float
    confidence: Optional[float] = None
    price_range: Optional[Dict[str, float]] = None
    factors: Optional[List[str]] = None


class ModelInfo(BaseModel):
    model_type: str
    is_trained: bool
    ml_available: bool
    features: Optional[List[str]] = None
    training_date: Optional[str] = None
    performance_metrics: Optional[Dict] = None


@app.get("/")
async def root():
    return {
        "message": "Real Estate Valuation API", 
        "status": "running",
        "model_type": "ML" if ML_AVAILABLE else "Lightweight"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": valuation_model.is_trained
    }


@app.post("/api/valuation", response_model=ValuationResponse)
async def predict_valuation(property_data: PropertyData):
    """
    不動産価格の査定を実行
    
    Args:
        property_data: 物件情報（都道府県、市区町村、地区、土地面積、建物面積、築年数）
    
    Returns:
        査定結果（推定価格、信頼度、価格範囲、価格要因）
    """
    try:
        logger.info(f"Received valuation request: {property_data}")
        
        result = valuation_model.predict(
            prefecture=property_data.prefecture,
            city=property_data.city,
            district=property_data.district,
            land_area=property_data.land_area,
            building_area=property_data.building_area,
            building_age=property_data.building_age
        )
        
        logger.info(f"Valuation result: {result}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during valuation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Valuation failed: {str(e)}")


@app.get("/api/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    現在のモデル情報を取得
    """
    try:
        info = {
            "model_type": "LightGBM" if ML_AVAILABLE else "Rule-based",
            "is_trained": valuation_model.is_trained,
            "ml_available": ML_AVAILABLE,
            "features": getattr(valuation_model, 'feature_columns', None)
        }
        
        # MLモデルの場合は詳細情報を追加
        if ML_AVAILABLE and hasattr(valuation_model, 'get_model_info'):
            model_details = valuation_model.get_model_info()
            info.update({
                "training_date": model_details.get('training_date'),
                "performance_metrics": {
                    "test_mae": model_details.get('test_metrics', {}).get('mae'),
                    "test_r2": model_details.get('test_metrics', {}).get('r2'),
                    "cv_mae": model_details.get('cv_scores', {}).get('mean')
                }
            })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post("/api/model/train")
async def train_model(
    fetch_new_data: bool = False,
    data_path: Optional[str] = None
):
    """
    モデルの訓練を実行（MLモデルのみ）
    
    Args:
        fetch_new_data: 新しいデータをMLIT APIから取得するか
        data_path: 既存のCSVファイルパス（指定しない場合は新規取得）
    """
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="Training is not available in lightweight mode"
        )
    
    try:
        logger.info("Starting model training...")
        
        results = valuation_model.train_model(
            data_path=data_path,
            fetch_new_data=fetch_new_data
        )
        
        return {
            "status": "success",
            "message": "Model training completed",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@app.get("/api/model/feature-importance")
async def get_feature_importance():
    """
    特徴量の重要度を取得（MLモデルのみ）
    """
    if not ML_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="Feature importance is not available in lightweight mode"
        )
    
    try:
        if hasattr(valuation_model, 'ml_model') and valuation_model.ml_model:
            if hasattr(valuation_model.ml_model, 'model_info'):
                feature_importance = valuation_model.ml_model.model_info.get('feature_importance', [])
                return {
                    "feature_importance": feature_importance[:10],  # Top 10
                    "total_features": len(feature_importance)
                }
        
        return {
            "feature_importance": [],
            "message": "Model not trained yet"
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)