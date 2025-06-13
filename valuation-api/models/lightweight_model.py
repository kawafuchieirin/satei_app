"""
機械学習モデル専用査定システム
MLモデルが必須
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
                logger.info("ML model files not found, using rule-based calculation")
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
    
    def _rule_based_predict(self, prefecture: str, city: str, district: str,
                           land_area: float, building_area: float, building_age: int) -> float:
        """ルールベースによる査定価格計算"""
        
        # 東京23区の基準単価（万円/㎡）
        tokyo_ward_prices = {
            '千代田区': 220, '中央区': 180, '港区': 200, '新宿区': 120, '文京区': 110,
            '台東区': 90, '墨田区': 75, '江東区': 85, '品川区': 100, '目黒区': 130,
            '大田区': 80, '世田谷区': 95, '渋谷区': 160, '中野区': 85, '杉並区': 90,
            '豊島区': 85, '北区': 70, '荒川区': 65, '板橋区': 70, '練馬区': 75,
            '足立区': 60, '葛飾区': 60, '江戸川区': 65
        }
        
        # 基準価格取得（デフォルト80万円/㎡）
        base_price_per_sqm = tokyo_ward_prices.get(city, 80)
        
        # 土地価格計算
        land_price = land_area * base_price_per_sqm
        
        # 建物価格計算（建築費50万円/㎡から減価償却）
        building_unit_cost = 50  # 万円/㎡
        depreciation_rate = min(0.03 * building_age, 0.70)  # 年3%、最大70%
        building_price = building_area * building_unit_cost * (1 - depreciation_rate)
        
        # 総額計算
        total_price = land_price + building_price
        
        # 立地補正
        if city in ['千代田区', '港区', '渋谷区']:
            total_price *= 1.2  # 一等地補正
        elif city in ['中央区', '新宿区', '文京区', '目黒区']:
            total_price *= 1.1  # 準一等地補正
        
        # 規模補正
        if land_area >= 150:
            total_price *= 1.1  # 大規模補正
        elif land_area <= 50:
            total_price *= 0.95  # 小規模補正
            
        logger.info(f"Rule-based calculation: ¥{total_price:,.0f}")
        return total_price
    
    def predict(self, prefecture: str, city: str, district: str,
                land_area: float, building_area: float, building_age: int) -> Dict:
        """
        査定価格を予測（MLモデルまたはルールベース）
        """
        try:
            # MLモデルが利用可能な場合はMLモデルを使用
            if self.ml_available and self.ml_model is not None:
                estimated_price = self._ml_predict(
                    prefecture, city, district, land_area, building_area, building_age
                )
                confidence = max(75, min(95, 90 - building_age * 0.3))
                method = "機械学習モデルによる高精度予測"
                logger.info("Using ML model prediction")
            else:
                # MLモデルが利用できない場合はルールベース計算
                estimated_price = self._rule_based_predict(
                    prefecture, city, district, land_area, building_area, building_age
                )
                confidence = max(60, min(85, 80 - building_age * 0.5))
                method = "ルールベースによる基本査定"
                logger.info("Using rule-based prediction")
            
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
    
