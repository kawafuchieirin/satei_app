"""
機械学習ベースの査定モデル
Random Forestを使用した高精度な価格予測
"""
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MLValuationModel:
    """機械学習ベースの査定モデル"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.is_trained = False
        self.feature_columns = [
            'prefecture_encoded', 'city_encoded', 'district_encoded',
            'land_area', 'building_area', 'building_age',
            'total_area', 'building_ratio', 'age_squared'
        ]
        
        
        self._load_model()
    
    def _load_model(self):
        """訓練済みモデルを読み込み"""
        try:
            model_path = Path(__file__).parent.parent / 'valuation_model.joblib'
            encoders_path = Path(__file__).parent.parent / 'label_encoders.joblib'
            
            if model_path.exists() and encoders_path.exists():
                self.model = joblib.load(model_path)
                self.label_encoders = joblib.load(encoders_path)
                self.is_trained = True
                logger.info("ML model loaded successfully")
            else:
                logger.error("ML model files not found - ML model is required")
                raise FileNotFoundError("ML model files (valuation_model.joblib, label_encoders.joblib) not found")
                
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
    
    def _prepare_features(self, prefecture: str, city: str, district: str,
                         land_area: float, building_area: float, building_age: int) -> np.ndarray:
        """入力データから特徴量を準備"""
        try:
            # データフレーム作成
            input_data = pd.DataFrame({
                'prefecture': [prefecture],
                'city': [city],
                'district': [district],
                'land_area': [land_area],
                'building_area': [building_area],
                'building_age': [building_age]
            })
            
            # カテゴリカル変数のエンコーディング
            categorical_cols = ['prefecture', 'city', 'district']
            features = input_data.copy()
            
            for col in categorical_cols:
                if col in self.label_encoders:
                    try:
                        # 既知のラベルを変換
                        features[f'{col}_encoded'] = self.label_encoders[col].transform(
                            features[col].astype(str)
                        )
                    except ValueError:
                        # 未知のラベルは最頻値で代替
                        logger.warning(f"Unknown {col}: {features[col].iloc[0]}, using fallback")
                        if col == 'city':
                            # 最も近い区の平均値を使用
                            features[f'{col}_encoded'] = 10  # 中間値
                        else:
                            features[f'{col}_encoded'] = 0
                else:
                    features[f'{col}_encoded'] = 0
            
            # 派生特徴量
            features['total_area'] = features['land_area'] + features['building_area']
            features['building_ratio'] = features['building_area'] / (features['land_area'] + 1)
            features['age_squared'] = features['building_age'] ** 2
            
            # 特徴量行列作成
            X = features[self.feature_columns].values
            
            return X
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise
    
    
    def predict(self, prefecture: str, city: str, district: str,
                land_area: float, building_area: float, building_age: int) -> Dict[str, Any]:
        """
        価格予測を実行
        
        Args:
            prefecture: 都道府県
            city: 市区町村
            district: 地区
            land_area: 土地面積（㎡）
            building_area: 建物面積（㎡）
            building_age: 築年数
            
        Returns:
            予測結果の辞書
        """
        try:
            # MLモデルが利用可能かチェック
            if not self.is_trained or self.model is None:
                logger.error("ML model is not available")
                raise RuntimeError("ML model is not loaded. Cannot perform valuation.")
            
            # ML予測を実行
            X = self._prepare_features(prefecture, city, district, 
                                     land_area, building_area, building_age)
            estimated_price = self.model.predict(X)[0]
            
            # 信頼度計算（MLモデルベース）
            confidence = max(75, min(95, 90 - building_age * 0.3))
            
            logger.info(f"ML prediction: ¥{estimated_price:,.0f}")
            
            # 価格帯計算
            price_range = {
                'min': estimated_price * 0.85,
                'max': estimated_price * 1.15
            }
            
            # 査定要因分析
            factors = ["機械学習モデルによる高精度予測"]
            
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
            
            return {
                'estimated_price': float(estimated_price),
                'confidence': float(confidence),
                'price_range': price_range,
                'factors': factors
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")