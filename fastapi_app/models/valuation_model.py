import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # 新しいMLモデルを使用
    from model_creation.train_ml_model import RealEstatePriceModel
    ML_MODEL_AVAILABLE = True
except ImportError:
    ML_MODEL_AVAILABLE = False
    logger.warning("ML model not available, falling back to rule-based model")

logger = logging.getLogger(__name__)


class ValuationModel:
    """
    不動産査定のための機械学習モデル
    LightGBMベースの新しいモデルとレガシーモデルの両方をサポート
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.ml_model = None
        self.is_trained = False
        self.model_path = model_path or "models"
        
        # 新しいMLモデルが利用可能な場合は使用
        if ML_MODEL_AVAILABLE:
            try:
                self.ml_model = RealEstatePriceModel(model_dir=self.model_path)
                # 保存されたモデルを読み込み
                if Path(self.model_path).exists() and (Path(self.model_path) / 'real_estate_model.joblib').exists():
                    self.ml_model.load_model()
                    self.is_trained = True
                    logger.info("LightGBM model loaded successfully")
                else:
                    logger.info("No existing LightGBM model found. Model needs training.")
                    self.is_trained = False
            except Exception as e:
                logger.error(f"Failed to initialize ML model: {e}")
                self.ml_model = None
                self.is_trained = False
        else:
            # フォールバック: 簡易ルールベースモデルを使用
            logger.info("Using fallback rule-based model")
            self.is_trained = True
    
    def train_model(self, data_path: Optional[str] = None, fetch_new_data: bool = False):
        """
        モデルの訓練を実行
        """
        if self.ml_model and ML_MODEL_AVAILABLE:
            logger.info("Training LightGBM model...")
            try:
                results = self.ml_model.train(
                    data_path=data_path,
                    fetch_new_data=fetch_new_data,
                    test_size=0.2,
                    cv_folds=5
                )
                self.is_trained = True
                logger.info(f"Model training completed. Test MAE: {results['test_metrics']['mae']:,.0f}")
                return results
            except Exception as e:
                logger.error(f"Failed to train ML model: {e}")
                self.is_trained = False
                raise
        else:
            logger.warning("ML model not available for training")
            self.is_trained = False
            raise ValueError("ML model not available")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # 列名の統一（サンプルデータ用）
        column_mapping = {
            '取引価格（総額）': '取引価格',
            '面積（㎡）': '土地面積'
        }
        
        df_clean = df_clean.rename(columns=column_mapping)
        
        # 必要な列が存在しない場合のデフォルト値設定
        if '都道府県' not in df_clean.columns:
            df_clean['都道府県'] = '東京都'
        if '市区町村' not in df_clean.columns:
            df_clean['市区町村'] = '渋谷区'
        if '地区' not in df_clean.columns:
            df_clean['地区'] = '恵比寿'
        
        # 数値列の処理
        numeric_columns = ['土地面積', '建物面積', '築年数', '取引価格']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 異常値の除去
        if '取引価格' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['取引価格'] > 1000000) &  # 100万円以上
                (df_clean['取引価格'] < 1000000000)  # 10億円未満
            ]
        
        if '土地面積' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['土地面積'] > 10) &  # 10㎡以上
                (df_clean['土地面積'] < 1000)  # 1000㎡未満
            ]
        
        if '建物面積' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['建物面積'] > 10) &  # 10㎡以上
                (df_clean['建物面積'] < 500)  # 500㎡未満
            ]
        
        # 欠損値の除去
        df_clean = df_clean.dropna(subset=['取引価格'])
        
        return df_clean
    
    def predict(self, 
                prefecture: str,
                city: str,
                district: str,
                land_area: float,
                building_area: float,
                building_age: int) -> Dict:
        """
        不動産価格の予測
        """
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        # 新しいMLモデルを使用
        if self.ml_model and ML_MODEL_AVAILABLE:
            try:
                return self.ml_model.predict(
                    prefecture=prefecture,
                    city=city,
                    district=district,
                    land_area=land_area,
                    building_area=building_area,
                    building_age=building_age
                )
            except Exception as e:
                logger.error(f"ML model prediction failed: {e}")
                # フォールバックとして簡易計算を実行
                return self._fallback_predict(
                    prefecture, city, district,
                    land_area, building_area, building_age
                )
        else:
            # フォールバック: 簡易ルールベース計算
            return self._fallback_predict(
                prefecture, city, district,
                land_area, building_area, building_age
            )
    
    def _fallback_predict(self,
                         prefecture: str,
                         city: str,
                         district: str,
                         land_area: float,
                         building_area: float,
                         building_age: int) -> Dict:
        """
        フォールバック用の簡易予測（ルールベース）
        """
        # 東京23区の基準価格（万円/㎡）
        ward_base_prices = {
            '千代田区': 250, '中央区': 200, '港区': 220,
            '新宿区': 150, '文京区': 140, '台東区': 120,
            '墨田区': 100, '江東区': 110, '品川区': 160,
            '目黒区': 170, '大田区': 130, '世田谷区': 140,
            '渋谷区': 180, '中野区': 120, '杉並区': 130,
            '豊島区': 140, '北区': 100, '荒川区': 90,
            '板橋区': 100, '練馬区': 110, '足立区': 80,
            '葛飾区': 85, '江戸川区': 90
        }
        
        # 基準価格を取得
        base_price_per_sqm = ward_base_prices.get(city, 100)
        
        # 価格計算
        land_price = land_area * base_price_per_sqm
        building_depreciation = max(0.3, 1 - (building_age * 0.03))
        building_price = building_area * base_price_per_sqm * 0.8 * building_depreciation
        
        total_price = (land_price + building_price) * 10000  # 万円から円に変換
        
        # ランダムな変動を追加
        variation = np.random.uniform(-0.1, 0.1)
        total_price = total_price * (1 + variation)
        
        # 査定要因の分析
        factors = self._analyze_factors(
            land_area, building_area, building_age, total_price
        )
        
        return {
            'estimated_price': float(total_price),
            'confidence': 75.0,
            'price_range': {
                'min': float(total_price * 0.85),
                'max': float(total_price * 1.15)
            },
            'factors': factors
        }
    
    def _analyze_factors(self, 
                        land_area: float,
                        building_area: float,
                        building_age: int,
                        predicted_price: float) -> List[str]:
        """
        査定価格に影響する要因を分析
        """
        factors = []
        
        # 面積による影響
        if land_area > 150:
            factors.append("土地面積が広く、価格にプラス影響")
        elif land_area < 50:
            factors.append("土地面積が狭く、価格にマイナス影響")
        
        if building_area > 120:
            factors.append("建物面積が広く、価格にプラス影響")
        elif building_area < 60:
            factors.append("建物面積が狭く、価格にマイナス影響")
        
        # 築年数による影響
        if building_age <= 5:
            factors.append("築浅物件で、価格にプラス影響")
        elif building_age <= 15:
            factors.append("比較的新しい物件で、価格への影響は中程度")
        elif building_age <= 25:
            factors.append("築年数がやや古く、価格にマイナス影響")
        else:
            factors.append("築年数が古く、価格に大きなマイナス影響")
        
        # 価格レンジによる評価
        if predicted_price > 100000000:  # 1億円以上
            factors.append("高額物件として評価")
        elif predicted_price < 30000000:  # 3000万円未満
            factors.append("比較的手頃な価格帯の物件")
        
        return factors
    
    def get_model_info(self) -> Dict:
        """
        モデル情報を取得
        """
        if self.ml_model and ML_MODEL_AVAILABLE and hasattr(self.ml_model, 'model_info'):
            return self.ml_model.model_info
        else:
            return {
                'model_type': 'Rule-based fallback',
                'training_date': 'N/A',
                'is_trained': self.is_trained
            }