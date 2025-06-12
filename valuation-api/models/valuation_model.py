import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import logging
from typing import Dict, List, Optional, Tuple

from .data_fetcher import MLITDataFetcher

logger = logging.getLogger(__name__)


class ValuationModel:
    """
    不動産査定のための機械学習モデル
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.encoders = {}
        self.feature_columns = [
            '都道府県', '市区町村', '地区', '土地面積', '建物面積', '築年数'
        ]
        self.is_trained = False
        self.model_path = model_path or "valuation_model.joblib"
        self.encoders_path = "label_encoders.joblib"
        
        # モデルが存在する場合は読み込み
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            logger.info("No existing model found. Training new model...")
            self.train_model()
    
    def train_model(self):
        """
        モデルの訓練を実行
        """
        logger.info("Starting model training...")
        
        # データ取得
        data_fetcher = MLITDataFetcher()
        
        # 実際のMLIT APIからデータを取得（失敗した場合はサンプルデータを使用）
        try:
            df = data_fetcher.fetch_trade_data(
                prefecture="東京都",
                from_year=2021,
                to_year=2024
            )
            
            if df.empty:
                logger.warning("No data from MLIT API, using sample data")
                df = data_fetcher.generate_sample_data()
        except Exception as e:
            logger.warning(f"Failed to fetch MLIT data: {e}, using sample data")
            df = data_fetcher.generate_sample_data()
        
        # データ前処理
        df_processed = self._preprocess_data(df)
        
        if df_processed.empty:
            raise ValueError("No valid data available for training")
        
        # 特徴量とターゲットの分離
        X = df_processed[self.feature_columns[:-1]]  # 築年数以外
        X['築年数'] = df_processed['築年数']
        y = df_processed['取引価格']
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['都道府県', '市区町村', '地区']
        for col in categorical_columns:
            if col in X.columns:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder
        
        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # モデル訓練
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # モデル評価
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model training completed. MAE: {mae:.2f}, R2: {r2:.3f}")
        
        self.is_trained = True
        
        # モデル保存
        self.save_model()
    
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
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained")
        
        # 入力データの準備
        input_data = pd.DataFrame({
            '都道府県': [prefecture],
            '市区町村': [city],
            '地区': [district],
            '土地面積': [land_area],
            '建物面積': [building_area],
            '築年数': [building_age]
        })
        
        # カテゴリカル変数のエンコーディング
        for col in ['都道府県', '市区町村', '地区']:
            if col in self.encoders:
                try:
                    input_data[col] = self.encoders[col].transform(input_data[col])
                except ValueError:
                    # 未知のカテゴリの場合は最頻値を使用
                    input_data[col] = 0
        
        # 予測実行
        predicted_price = self.model.predict(input_data)[0]
        
        # 信頼区間の計算（簡易版）
        if hasattr(self.model, 'estimators_'):
            predictions = np.array([
                estimator.predict(input_data)[0] 
                for estimator in self.model.estimators_
            ])
            std_pred = np.std(predictions)
            confidence = max(0, min(100, 100 - (std_pred / predicted_price) * 100))
            
            price_range = {
                'min': max(0, predicted_price - 1.96 * std_pred),
                'max': predicted_price + 1.96 * std_pred
            }
        else:
            confidence = 75.0  # デフォルト値
            price_range = {
                'min': predicted_price * 0.8,
                'max': predicted_price * 1.2
            }
        
        # 査定要因の分析
        factors = self._analyze_factors(
            land_area, building_area, building_age, predicted_price
        )
        
        return {
            'estimated_price': float(predicted_price),
            'confidence': float(confidence),
            'price_range': price_range,
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
    
    def save_model(self):
        """
        モデルとエンコーダーを保存
        """
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        
        if self.encoders:
            joblib.dump(self.encoders, self.encoders_path)
            logger.info(f"Encoders saved to {self.encoders_path}")
    
    def load_model(self):
        """
        保存されたモデルとエンコーダーを読み込み
        """
        try:
            self.model = joblib.load(self.model_path)
            self.encoders = joblib.load(self.encoders_path)
            self.is_trained = True
            logger.info("Model and encoders loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.is_trained = False