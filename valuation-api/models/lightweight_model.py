"""
機械学習モデル専用査定システム
MLモデルが必須 - ルールベース査定は削除済み
"""
import json
from typing import Dict, List, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# MLモデルの動的インポート（Lambda環境では利用不可）
try:
    import joblib
    import pandas as pd
    import numpy as np
    ML_AVAILABLE = True
    logger.info("ML dependencies available")
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML dependencies not available")

class LightweightValuationModel:
    """機械学習モデル専用査定システム"""
    
    def __init__(self):
        self.is_trained = True
        self.feature_columns = ['prefecture', 'city', 'district', 'land_area', 'building_area', 'building_age']
        self.ml_model = None
        self.ml_encoders = None
        self.ml_available = False
        
        
        # MLモデルの読み込みを試行
        if ML_AVAILABLE:
            self._try_load_ml_model()
        
        logger.info("Lightweight valuation model initialized")
    
    def _try_load_ml_model(self):
        """MLモデルの読み込みを試行"""
        try:
            model_path = Path(__file__).parent.parent / 'valuation_model.joblib'
            encoders_path = Path(__file__).parent.parent / 'label_encoders.joblib'
            
            if model_path.exists() and encoders_path.exists():
                self.ml_model = joblib.load(model_path)
                self.ml_encoders = joblib.load(encoders_path)
                self.ml_available = True
                logger.info("ML model loaded successfully")
            else:
                logger.error("ML model files not found - ML model is required")
                raise FileNotFoundError("ML model files (valuation_model.joblib, label_encoders.joblib) not found")
        except Exception as e:
            logger.warning(f"Failed to load ML model: {e}")
    
    def _ml_predict(self, prefecture: str, city: str, district: str,
                   land_area: float, building_area: float, building_age: int) -> float:
        """MLモデルによる予測"""
        try:
            # 入力データフレーム作成
            input_data = pd.DataFrame({
                'prefecture': [prefecture],
                'city': [city], 
                'district': [district],
                'land_area': [land_area],
                'building_area': [building_area],
                'building_age': [building_age]
            })
            
            # カテゴリカル変数のエンコーディング
            features = input_data.copy()
            categorical_cols = ['prefecture', 'city', 'district']
            
            for col in categorical_cols:
                if col in self.ml_encoders:
                    try:
                        features[f'{col}_encoded'] = self.ml_encoders[col].transform(
                            features[col].astype(str)
                        )
                    except ValueError:
                        # 未知のラベルは中間値で代替
                        logger.warning(f"Unknown {col}: {features[col].iloc[0]}")
                        features[f'{col}_encoded'] = 10 if col == 'city' else 0
                else:
                    features[f'{col}_encoded'] = 0
            
            # 派生特徴量
            features['total_area'] = features['land_area'] + features['building_area']
            features['building_ratio'] = features['building_area'] / (features['land_area'] + 1)
            features['age_squared'] = features['building_age'] ** 2
            
            # 特徴量選択
            feature_cols = [
                'prefecture_encoded', 'city_encoded', 'district_encoded',
                'land_area', 'building_area', 'building_age',
                'total_area', 'building_ratio', 'age_squared'
            ]
            
            X = features[feature_cols]
            
            # 予測実行
            predicted_price = self.ml_model.predict(X)[0]
            
            logger.info(f"ML prediction: ¥{predicted_price:,.0f}")
            return predicted_price
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            raise
    
    
    def predict(self, prefecture: str, city: str, district: str,
                land_area: float, building_area: float, building_age: int) -> Dict:
        """
        査定価格を予測（MLモデル専用）
        """
        try:
            # MLモデルが利用可能かチェック
            if not self.ml_available or self.ml_model is None:
                logger.error("ML model is not available")
                raise RuntimeError("ML model is not loaded. Cannot perform valuation.")
            
            # MLモデルで予測実行
            estimated_price = self._ml_predict(
                prefecture, city, district, land_area, building_area, building_age
            )
            confidence = max(75, min(95, 90 - building_age * 0.3))
            method = "機械学習モデルによる高精度予測"
            logger.info("Using ML model prediction")
            
            # 価格帯計算
            price_range = {
                'min': estimated_price * 0.85,
                'max': estimated_price * 1.15
            }
            
            # 査定要因分析
            factors = [method]
            
            if building_age <= 5:
                factors.append("築浅物件で、価格にプラス影響")
            elif building_age <= 15:
                factors.append("比較的新しい物件で、価格への影響は中程度")
            else:
                factors.append(f"築{building_age}年で、建物価値が減少")
                
            if land_area >= 100:
                factors.append("土地面積が広く、価格にプラス影響")
                
            if city in ['千代田区', '港区', '渋谷区', '中央区']:
                factors.append("都心の一等地で、価格が高め")
            
            logger.info(f"Prediction completed: ¥{estimated_price:,.0f}")
            
            return {
                'estimated_price': float(estimated_price),
                'confidence': float(confidence),
                'price_range': price_range,
                'factors': factors
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
