from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from typing import Optional, List, Dict
import os
import sys
from pathlib import Path

# モデルクリエーションディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent.parent / 'model-creation'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ML専用モデルの読み込み
try:
    from train_xgboost_model import MultiModelTrainer
    trainer = MultiModelTrainer(model_dir=str(Path(__file__).parent / 'models'))
    
    # 最良モデルのロード試行
    if not trainer.load_best_model():
        logger.warning("No pre-trained model found. Please train the model first.")
        # デモ用の簡易モデルを使用
        from models.lightweight_model import LightweightValuationModel
        valuation_model = LightweightValuationModel()
    else:
        logger.info("Successfully loaded best ML model")
        valuation_model = trainer
        
except Exception as e:
    logger.error(f"Failed to load ML model: {e}")
    # フォールバックとして既存のlightweight_modelを使用
    from models.lightweight_model import LightweightValuationModel
    valuation_model = LightweightValuationModel()

app = FastAPI(
    title="Real Estate Valuation API (ML-Only)",
    description="不動産価格査定API - 機械学習モデル専用版",
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
    district: str
    land_area: float
    building_area: float
    building_age: int


class ValuationResponse(BaseModel):
    estimated_price: float
    confidence: Optional[float] = None
    price_range: Optional[Dict[str, float]] = None
    factors: Optional[List[str]] = None
    model_type: Optional[str] = None
    model_metrics: Optional[Dict] = None


class ModelTrainingRequest(BaseModel):
    fetch_new_data: bool = False
    test_size: float = 0.2
    cv_folds: int = 5


class ModelComparisonResponse(BaseModel):
    models: Dict[str, Dict]
    best_model: str
    training_date: str


@app.get("/")
async def root():
    return {
        "message": "Real Estate Valuation API (ML-Only)", 
        "status": "running",
        "version": "2.0.0",
        "model_loaded": hasattr(valuation_model, 'best_model') and valuation_model.best_model is not None
    }


@app.get("/health")
async def health_check():
    model_status = "loaded" if hasattr(valuation_model, 'best_model') and valuation_model.best_model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status
    }


@app.post("/api/valuation", response_model=ValuationResponse)
async def predict_valuation(property_data: PropertyData):
    """
    不動産価格の査定（ML専用）
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
        
        # MultiModelTrainerの場合、追加情報を含める
        if hasattr(valuation_model, 'model_info'):
            result['model_type'] = valuation_model.model_info.get('model_type', 'unknown')
            result['model_metrics'] = valuation_model.model_info.get('metrics', {})
        
        logger.info(f"Valuation result: {result}")
        return ValuationResponse(**result)
        
    except Exception as e:
        logger.error(f"Error during valuation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Valuation failed: {str(e)}")


@app.post("/api/model/train")
async def train_models(request: ModelTrainingRequest):
    """
    全モデルの訓練と比較を実行
    """
    try:
        logger.info("Starting model training...")
        
        # 新しいトレーナーインスタンスを作成
        new_trainer = MultiModelTrainer(model_dir=str(Path(__file__).parent / 'models'))
        
        # モデル訓練を実行
        results = new_trainer.train_all_models(
            fetch_new_data=request.fetch_new_data,
            test_size=request.test_size,
            cv_folds=request.cv_folds
        )
        
        # グローバルなvaluation_modelを更新
        global valuation_model
        valuation_model = new_trainer
        
        # 結果を整形
        response_data = {
            "models": {},
            "best_model": new_trainer.model_info.get('model_type', 'unknown'),
            "training_date": new_trainer.model_info.get('training_date', '')
        }
        
        for model_name, metrics in results.items():
            response_data["models"][model_name] = {
                "r2": metrics.get('r2', 0),
                "rmse": metrics.get('rmse', 0),
                "mae": metrics.get('mae', 0),
                "mape": metrics.get('mape', 0)
            }
        
        return ModelComparisonResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")


@app.get("/api/model/comparison")
async def get_model_comparison():
    """
    訓練済みモデルの比較結果を取得
    """
    try:
        import json
        model_dir = Path(__file__).parent / 'models'
        results_path = model_dir / 'model_comparison_results.json'
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="No model comparison results found. Please train models first.")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 最良モデル情報を追加
        best_model_info_path = model_dir / 'best_model_info.json'
        if best_model_info_path.exists():
            with open(best_model_info_path, 'r', encoding='utf-8') as f:
                best_info = json.load(f)
                results['best_model'] = best_info
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting model comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model comparison: {str(e)}")


@app.get("/api/model/info")
async def get_model_info():
    """
    現在のモデル情報を取得
    """
    try:
        info = {
            "model_loaded": hasattr(valuation_model, 'best_model') and valuation_model.best_model is not None,
            "model_type": "ML-Only",
            "ml_models_available": ["linear_regression", "ridge", "lasso", "random_forest", "xgboost", "lightgbm"]
        }
        
        # MultiModelTrainerの情報
        if hasattr(valuation_model, 'model_info'):
            info.update({
                "current_model": valuation_model.model_info.get('model_type', 'unknown'),
                "metrics": valuation_model.model_info.get('metrics', {}),
                "training_date": valuation_model.model_info.get('training_date', 'unknown')
            })
        
        # モデルファイルの存在確認
        model_dir = Path(__file__).parent / 'models'
        model_files = {
            "best_model": (model_dir / 'best_model.joblib').exists(),
            "linear_regression": (model_dir / 'linear_regression_model.joblib').exists(),
            "ridge": (model_dir / 'ridge_model.joblib').exists(),
            "lasso": (model_dir / 'lasso_model.joblib').exists(),
            "random_forest": (model_dir / 'random_forest_model.joblib').exists(),
            "xgboost": (model_dir / 'xgboost_model.joblib').exists(),
            "lightgbm": (model_dir / 'lightgbm_model.joblib').exists()
        }
        info["model_files"] = model_files
        
        # データ前処理ファイルの確認
        preprocessor_files = {
            "label_encoders": (model_dir / 'label_encoders.joblib').exists(),
            "scaler": (model_dir / 'scaler.joblib').exists()
        }
        info["preprocessor_files"] = preprocessor_files
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.post("/api/model/evaluate")
async def evaluate_current_model():
    """
    現在のモデルを評価
    """
    try:
        if not (hasattr(valuation_model, 'best_model') and valuation_model.best_model is not None):
            raise HTTPException(status_code=400, detail="No trained model available for evaluation")
        
        # 評価用のデータを読み込む
        from pathlib import Path
        import pandas as pd
        
        data_dir = Path(__file__).parent.parent / 'model-creation' / 'data'
        data_files = list(data_dir.glob('tokyo23_real_estate_*.csv'))
        
        if not data_files:
            raise HTTPException(status_code=404, detail="No evaluation data found")
        
        # 最新のデータファイルを使用
        latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
        
        # モデル評価を実行
        from model_evaluator import evaluate_models_from_file
        results, report = evaluate_models_from_file(
            str(latest_data), 
            str(Path(__file__).parent / 'models')
        )
        
        return {
            "evaluation_report": report,
            "detailed_results": results
        }
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


@app.get("/api/data/fetch")
async def fetch_mlit_data(from_year: int = 2022, to_year: int = 2024):
    """
    MLITから最新データを取得
    """
    try:
        from tokyo23_data_fetcher import Tokyo23DataFetcher
        
        fetcher = Tokyo23DataFetcher()
        logger.info(f"Fetching MLIT data from {from_year} to {to_year}...")
        
        # データ取得（非同期ではないため、長時間かかる可能性あり）
        df = fetcher.fetch_all_tokyo23_data(
            from_year=from_year,
            to_year=to_year,
            save_csv=True
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data fetched from MLIT API")
        
        # 統計情報を取得
        stats = fetcher.get_summary_statistics(df)
        
        return {
            "message": "Data fetched successfully",
            "statistics": stats,
            "from_year": from_year,
            "to_year": to_year
        }
        
    except Exception as e:
        logger.error(f"Error fetching MLIT data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data fetching failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)