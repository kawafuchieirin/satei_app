from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List, Dict
import os

from models.valuation_model import ValuationModel
from models.model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real Estate Valuation API",
    description="API for real estate property valuation using ML models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

valuation_model = ValuationModel()
model_evaluator = None  # Lazy loading for evaluation


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


class ModelEvaluationResponse(BaseModel):
    overall_metrics: Dict
    price_range_accuracy: Dict
    feature_importance: Dict
    evaluation_summary: List[str]
    sample_count: int


class CrossValidationResponse(BaseModel):
    r2_scores: List[float]
    mae_scores: List[float]
    r2_mean: float
    r2_std: float
    mae_mean: float
    mae_std: float
    cv_folds: int


@app.get("/")
async def root():
    return {"message": "Real Estate Valuation API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/valuation", response_model=ValuationResponse)
async def predict_valuation(property_data: PropertyData):
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
        
    except Exception as e:
        logger.error(f"Error during valuation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Valuation failed: {str(e)}")


@app.get("/api/model/evaluate", response_model=ModelEvaluationResponse)
async def evaluate_model():
    """
    モデルの性能評価を実行
    """
    global model_evaluator
    
    try:
        logger.info("Starting model evaluation...")
        
        if model_evaluator is None:
            model_evaluator = ModelEvaluator()
            model_evaluator.load_model_and_data()
        
        evaluation_results = model_evaluator.evaluate_model_performance()
        
        response = ModelEvaluationResponse(
            overall_metrics=evaluation_results['overall_metrics'],
            price_range_accuracy=evaluation_results['price_range_accuracy'],
            feature_importance=evaluation_results['feature_importance'],
            evaluation_summary=evaluation_results['evaluation_summary'],
            sample_count=evaluation_results['sample_count']
        )
        
        logger.info("Model evaluation completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


@app.get("/api/model/cross-validate", response_model=CrossValidationResponse)
async def cross_validate_model(cv_folds: int = 5):
    """
    モデルの交差検証を実行
    """
    global model_evaluator
    
    try:
        logger.info(f"Starting {cv_folds}-fold cross validation...")
        
        if model_evaluator is None:
            model_evaluator = ModelEvaluator()
            model_evaluator.load_model_and_data()
        
        cv_results = model_evaluator.cross_validate_model(cv_folds=cv_folds)
        
        response = CrossValidationResponse(**cv_results)
        
        logger.info("Cross validation completed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error during cross validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cross validation failed: {str(e)}")


@app.get("/api/model/prediction-samples")
async def get_prediction_samples(n_samples: int = 10):
    """
    予測サンプルを取得（実際の取引データとの比較）
    """
    global model_evaluator
    
    try:
        logger.info(f"Generating {n_samples} prediction samples...")
        
        if model_evaluator is None:
            model_evaluator = ModelEvaluator()
            model_evaluator.load_model_and_data()
        
        samples = model_evaluator.generate_prediction_samples(n_samples=n_samples)
        
        logger.info(f"Generated {len(samples)} prediction samples")
        return {"samples": samples, "count": len(samples)}
        
    except Exception as e:
        logger.error(f"Error generating prediction samples: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction sample generation failed: {str(e)}")


@app.post("/api/model/retrain")
async def retrain_model(model_type: str = "rf", data_source: str = "sample"):
    """
    モデルの再学習を実行
    """
    try:
        logger.info(f"Starting model retraining with type={model_type}, data_source={data_source}")
        
        # 別プロセスで学習を実行（非同期）
        import subprocess
        import os
        
        script_path = os.path.join(os.path.dirname(__file__), "train_model.py")
        cmd = [
            "python", script_path,
            "--model-type", model_type,
            "--data-source", data_source,
            "--output-dir", "."
        ]
        
        # バックグラウンドで実行
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 短時間待機して初期エラーをチェック
        try:
            stdout, stderr = process.communicate(timeout=5)
            if process.returncode != 0:
                raise Exception(f"Training failed: {stderr}")
        except subprocess.TimeoutExpired:
            # まだ実行中の場合は正常
            pass
        
        return {
            "message": "Model retraining started",
            "model_type": model_type,
            "data_source": data_source,
            "status": "in_progress"
        }
        
    except Exception as e:
        logger.error(f"Error starting model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")


@app.get("/api/model/info")
async def get_model_info():
    """
    現在のモデル情報を取得
    """
    try:
        import os
        import json
        from pathlib import Path
        
        info = {
            "model_loaded": valuation_model.is_trained,
            "model_type": "RandomForest",
            "features": valuation_model.feature_columns,
        }
        
        # 学習情報ファイルを探す
        training_info_files = list(Path(".").glob("training_info_*.json"))
        if training_info_files:
            # 最新のファイルを取得
            latest_file = max(training_info_files, key=os.path.getmtime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    training_info = json.load(f)
                info.update(training_info)
            except Exception as e:
                logger.warning(f"Failed to load training info: {e}")
        
        # モデルファイルの存在確認
        model_files = {
            "model_file": os.path.exists("valuation_model.joblib"),
            "encoders_file": os.path.exists("label_encoders.joblib"),
            "scaler_file": os.path.exists("scaler.joblib")
        }
        info["files"] = model_files
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)