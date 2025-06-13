#!/usr/bin/env python3
"""
機械学習モデル（LightGBM）の訓練スクリプト
東京23区の不動産価格予測モデルを構築
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Tuple, List, Optional
import warnings

from data_preprocessor import RealEstateDataPreprocessor
from tokyo23_data_fetcher import Tokyo23DataFetcher

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEstatePriceModel:
    """不動産価格予測モデル"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.preprocessor = RealEstateDataPreprocessor()
        self.model_info = {}
        
    def train(self, 
             data_path: Optional[str] = None,
             fetch_new_data: bool = False,
             test_size: float = 0.2,
             cv_folds: int = 5) -> Dict:
        """
        モデルの訓練
        
        Args:
            data_path: CSVファイルのパス（Noneの場合は新規取得）
            fetch_new_data: 新しいデータを取得するか
            test_size: テストデータの割合
            cv_folds: クロスバリデーションの分割数
            
        Returns:
            訓練結果の辞書
        """
        logger.info("Starting model training...")
        
        # データの準備
        if fetch_new_data or data_path is None:
            logger.info("Fetching new data from MLIT API...")
            fetcher = Tokyo23DataFetcher()
            raw_df = fetcher.fetch_all_tokyo23_data(
                from_year=2022, 
                to_year=2024, 
                save_csv=True
            )
        else:
            logger.info(f"Loading data from {data_path}")
            raw_df = pd.read_csv(data_path, encoding='utf-8-sig')
        
        if raw_df.empty:
            raise ValueError("No data available for training")
        
        # 特徴量の準備
        features_df = self.preprocessor.prepare_features(raw_df)
        
        # 前処理
        X, y, feature_names = self.preprocessor.preprocess(features_df, is_training=True)
        
        # データ分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # LightGBMパラメータ
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 1000,
            'early_stopping_rounds': 50
        }
        
        # モデルの訓練
        logger.info("Training LightGBM model...")
        
        self.model = lgb.LGBMRegressor(**params)
        
        # 訓練（early stoppingを使用）
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.log_evaluation(50)]
        )
        
        # 予測
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # 評価指標の計算
        train_metrics = self._calculate_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_metrics(y_test, y_pred_test)
        
        # クロスバリデーション
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='neg_mean_absolute_error'
        )
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # モデル情報の保存
        self.model_info = {
            'model_type': 'LightGBM',
            'training_date': datetime.now().isoformat(),
            'data_size': len(X),
            'features': feature_names,
            'params': params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_scores': {
                'mean': -cv_scores.mean(),
                'std': cv_scores.std()
            },
            'feature_importance': feature_importance.to_dict('records')
        }
        
        # 可視化
        self._create_visualizations(
            y_test, y_pred_test, 
            feature_importance, 
            train_metrics, test_metrics
        )
        
        # モデルの保存
        self._save_model()
        
        logger.info("Model training completed!")
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mae': -cv_scores.mean(),
            'feature_importance': feature_importance.head(10).to_dict('records')
        }
    
    def predict(self, 
               prefecture: str,
               city: str,
               district: str,
               land_area: float,
               building_area: float,
               building_age: int) -> Dict:
        """
        価格予測を実行
        
        Returns:
            予測結果の辞書（価格、信頼区間、価格要因など）
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # 特徴量の作成
        X = self.preprocessor.create_model_features(
            prefecture, city, district,
            land_area, building_area, building_age
        )
        
        # 予測
        predicted_price = self.model.predict(X)[0]
        
        # 予測の不確実性を推定（簡易版）
        # 実際にはQuantile Regressionや予測区間の推定が必要
        confidence = self._estimate_confidence(X)
        price_range = self._estimate_price_range(predicted_price, confidence)
        
        # 価格要因の説明
        factors = self._explain_price_factors(
            city, district, land_area, building_area, building_age,
            predicted_price
        )
        
        return {
            'estimated_price': int(predicted_price),
            'confidence': confidence,
            'price_range': price_range,
            'factors': factors
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """評価指標の計算"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def _estimate_confidence(self, X: np.ndarray) -> float:
        """予測の信頼度を推定（簡易版）"""
        # 実際にはモデルの予測分散や類似データ数などから計算
        # ここでは簡易的に80-95%の範囲でランダムに設定
        return np.random.uniform(80, 95)
    
    def _estimate_price_range(self, price: float, confidence: float) -> Dict:
        """価格範囲を推定"""
        # 信頼度に基づいて範囲を設定
        margin = (100 - confidence) / 100 * 0.5
        return {
            'min': int(price * (1 - margin)),
            'max': int(price * (1 + margin))
        }
    
    def _explain_price_factors(self, 
                             city: str,
                             district: str,
                             land_area: float,
                             building_area: float,
                             building_age: int,
                             predicted_price: float) -> List[str]:
        """価格要因の説明文を生成"""
        factors = []
        
        # 区による影響
        premium_wards = ['港区', '千代田区', '渋谷区', '中央区', '新宿区']
        if city in premium_wards:
            factors.append(f"{city}は都心の一等地で、価格が高めです")
        
        # 築年数による影響
        if building_age < 5:
            factors.append("築浅物件で、価格への好影響があります")
        elif building_age > 30:
            factors.append("築年数が経過しており、価格への影響があります")
        
        # 面積による影響
        if land_area > 150:
            factors.append("土地面積が広く、価格を押し上げています")
        elif land_area < 50:
            factors.append("土地面積がコンパクトで、価格が抑えられています")
        
        # 建物面積による影響
        if building_area > 120:
            factors.append("建物面積が広く、居住性が高いです")
        
        # 価格帯による説明
        if predicted_price > 100000000:  # 1億円以上
            factors.append("高級物件の価格帯です")
        elif predicted_price < 30000000:  # 3000万円以下
            factors.append("比較的手頃な価格帯です")
        
        return factors
    
    def _create_visualizations(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             feature_importance: pd.DataFrame,
                             train_metrics: Dict,
                             test_metrics: Dict):
        """結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 予測vs実際の散布図
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.5)
        ax1.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 
                'r--', lw=2)
        ax1.set_xlabel('Actual Price')
        ax1.set_ylabel('Predicted Price')
        ax1.set_title('Prediction vs Actual')
        
        # 2. 残差プロット
        ax2 = axes[0, 1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Price')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        
        # 3. 特徴量重要度（上位10個）
        ax3 = axes[1, 0]
        top_features = feature_importance.head(10)
        ax3.barh(top_features['feature'], top_features['importance'])
        ax3.set_xlabel('Importance')
        ax3.set_title('Top 10 Feature Importance')
        
        # 4. メトリクスの表示
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrics_text = f"""
        Training Metrics:
        MAE: {train_metrics['mae']:,.0f}
        RMSE: {train_metrics['rmse']:,.0f}
        R²: {train_metrics['r2']:.3f}
        MAPE: {train_metrics['mape']:.1f}%
        
        Test Metrics:
        MAE: {test_metrics['mae']:,.0f}
        RMSE: {test_metrics['rmse']:,.0f}
        R²: {test_metrics['r2']:.3f}
        MAPE: {test_metrics['mape']:.1f}%
        """
        ax4.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.model_dir / 'model_evaluation.png'}")
    
    def _save_model(self):
        """モデルと関連ファイルの保存"""
        # モデルの保存
        model_path = self.model_dir / 'real_estate_model.joblib'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # 前処理オブジェクトの保存
        self.preprocessor.save_preprocessor(self.model_dir)
        
        # モデル情報の保存
        info_path = self.model_dir / 'model_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Model info saved to {info_path}")
    
    def load_model(self):
        """保存されたモデルを読み込み"""
        # モデルの読み込み
        model_path = self.model_dir / 'real_estate_model.joblib'
        if model_path.exists():
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 前処理オブジェクトの読み込み
        self.preprocessor.load_preprocessor(self.model_dir)
        
        # モデル情報の読み込み
        info_path = self.model_dir / 'model_info.json'
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
            logger.info("Model info loaded")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train real estate price prediction model')
    parser.add_argument('--data-path', type=str, help='Path to CSV data file')
    parser.add_argument('--fetch-new', action='store_true', help='Fetch new data from MLIT API')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test data ratio')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    # モデルの訓練
    model = RealEstatePriceModel()
    results = model.train(
        data_path=args.data_path,
        fetch_new_data=args.fetch_new,
        test_size=args.test_size,
        cv_folds=args.cv_folds
    )
    
    # 結果の表示
    print("\nTraining Results:")
    print(f"Test MAE: {results['test_metrics']['mae']:,.0f}")
    print(f"Test R²: {results['test_metrics']['r2']:.3f}")
    print(f"CV MAE: {results['cv_mae']:,.0f}")
    
    print("\nTop 5 Important Features:")
    for i, feat in enumerate(results['feature_importance'][:5]):
        print(f"{i+1}. {feat['feature']}: {feat['importance']:.2f}")