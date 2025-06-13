#!/usr/bin/env python3
"""
シンプルなMLモデル学習スクリプト
Random Forestを使った不動産価格予測モデル
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRealEstateModel:
    """シンプルな不動産価格予測モデル"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_data(self, df: pd.DataFrame):
        """データの前処理"""
        logger.info("Preparing data...")
        
        # 特徴量選択
        features = df.copy()
        
        # カテゴリカル変数のエンコーディング
        categorical_cols = ['prefecture', 'city', 'district']
        for col in categorical_cols:
            if col in features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features[col].astype(str))
                else:
                    features[f'{col}_encoded'] = self.label_encoders[col].transform(features[col].astype(str))
        
        # 数値特徴量
        numerical_cols = ['land_area', 'building_area', 'building_age']
        feature_cols = [f'{col}_encoded' for col in categorical_cols if col in features.columns] + \
                      [col for col in numerical_cols if col in features.columns]
        
        # 派生特徴量
        if 'land_area' in features.columns and 'building_area' in features.columns:
            features['total_area'] = features['land_area'] + features['building_area']
            features['building_ratio'] = features['building_area'] / (features['land_area'] + 1)
            feature_cols.extend(['total_area', 'building_ratio'])
        
        if 'building_age' in features.columns:
            features['age_squared'] = features['building_age'] ** 2
            feature_cols.append('age_squared')
        
        X = features[feature_cols]
        y = features['trade_price']
        
        # 異常値除去
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        mask = (y >= lower) & (y <= upper)
        X = X[mask]
        y = y[mask]
        
        self.feature_columns = feature_cols
        logger.info(f"Features: {feature_cols}")
        logger.info(f"Data shape after cleaning: {X.shape}")
        
        return X, y
    
    def train(self, data_path: str):
        """モデル学習"""
        logger.info("Starting model training...")
        
        # データ読み込み
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records")
        
        # データ前処理
        X, y = self.prepare_data(df)
        
        # 訓練・テストデータ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data: {X_train.shape}")
        logger.info(f"Test data: {X_test.shape}")
        
        # モデル学習
        logger.info("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # 予測と評価
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # メトリクス計算
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        logger.info("=== Model Performance ===")
        logger.info(f"Training MAE: ¥{train_mae:,.0f}")
        logger.info(f"Test MAE: ¥{test_mae:,.0f}")
        logger.info(f"Training R²: {train_r2:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        
        # クロスバリデーション
        logger.info("Performing cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        
        logger.info(f"CV MAE: ¥{-cv_scores.mean():,.0f} (+/- ¥{cv_scores.std() * 2:,.0f})")
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("=== Feature Importance ===")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
    
    def save_model(self, model_dir: str = 'models'):
        """モデル保存"""
        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)
        
        # モデル保存
        joblib.dump(self.model, model_dir / 'real_estate_model.joblib')
        joblib.dump(self.label_encoders, model_dir / 'label_encoders.joblib')
        
        # API用にコピー
        api_model_dir = Path('../valuation-api')
        if api_model_dir.exists():
            joblib.dump(self.model, api_model_dir / 'valuation_model.joblib')
            joblib.dump(self.label_encoders, api_model_dir / 'label_encoders.joblib')
            logger.info("Models copied to API directory")
        
        logger.info(f"Models saved to {model_dir}")
    
    def predict(self, prefecture: str, city: str, district: str, 
                land_area: float, building_area: float, building_age: int):
        """価格予測"""
        # 入力データ準備
        input_data = pd.DataFrame({
            'prefecture': [prefecture],
            'city': [city],
            'district': [district],
            'land_area': [land_area],
            'building_area': [building_area],
            'building_age': [building_age],
            'trade_price': [0]  # ダミー値
        })
        
        # 前処理
        X, _ = self.prepare_data(input_data)
        
        # 予測
        predicted_price = self.model.predict(X)[0]
        
        # 信頼度計算（簡易版）
        confidence = max(70, min(95, 90 - building_age * 0.5))
        
        # 価格帯
        price_range = {
            'min': predicted_price * 0.85,
            'max': predicted_price * 1.15
        }
        
        return {
            'estimated_price': predicted_price,
            'confidence': confidence,
            'price_range': price_range,
            'factors': [f"機械学習モデル予測: ¥{predicted_price:,.0f}"]
        }

def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Real Estate Model Trainer')
    parser.add_argument('--data-path', default='data/tokyo23_sample_data.csv', 
                       help='Path to training data CSV')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='Save trained model')
    
    args = parser.parse_args()
    
    # モデル作成・学習
    model = SimpleRealEstateModel()
    results = model.train(args.data_path)
    
    if args.save_model:
        model.save_model()
    
    # テスト予測
    logger.info("=== Test Prediction ===")
    test_result = model.predict(
        prefecture='東京都',
        city='渋谷区', 
        district='恵比寿',
        land_area=100.0,
        building_area=80.0,
        building_age=10
    )
    logger.info(f"Test prediction: ¥{test_result['estimated_price']:,.0f}")

if __name__ == "__main__":
    main()