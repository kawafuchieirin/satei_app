#!/usr/bin/env python3
"""
XGBoostモデルの訓練スクリプト
東京23区の不動産価格予測モデルを構築
"""

import pandas as pd
import numpy as np
# import xgboost as xgb  # XGBoostが利用できない場合はコメントアウト
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModelTrainer:
    """複数の機械学習モデルを訓練・評価するクラス"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models = {}
        self.preprocessor = RealEstateDataPreprocessor()
        self.model_info = {}
        self.best_model = None
        
    def train_all_models(self, 
                        data_path: str = "data/tokyo23_real_estate.csv",
                        test_size: float = 0.2,
                        cv_folds: int = 5) -> Dict:
        """
        全モデルの訓練と評価
        
        Args:
            data_path: CSVファイルのパス
            test_size: テストデータの割合
            cv_folds: クロスバリデーションの分割数
            
        Returns:
            全モデルの評価結果
        """
        logger.info("Starting multi-model training...")
        
        # データの読み込み
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
        
        # 各モデルの訓練
        results = {}
        
        # 1. 線形回帰
        logger.info("Training Linear Regression...")
        results['linear_regression'] = self._train_linear_regression(
            X_train, X_test, y_train, y_test, cv_folds
        )
        
        # 2. Ridge回帰
        logger.info("Training Ridge Regression...")
        results['ridge'] = self._train_ridge(
            X_train, X_test, y_train, y_test, cv_folds
        )
        
        # 3. Lasso回帰
        logger.info("Training Lasso Regression...")
        results['lasso'] = self._train_lasso(
            X_train, X_test, y_train, y_test, cv_folds
        )
        
        # 4. Random Forest
        logger.info("Training Random Forest...")
        results['random_forest'] = self._train_random_forest(
            X_train, X_test, y_train, y_test, cv_folds
        )
        
        # 5. XGBoost（利用できない場合はスキップ）
        try:
            import xgboost as xgb
            logger.info("Training XGBoost...")
            results['xgboost'] = self._train_xgboost(
                X_train, X_test, y_train, y_test, cv_folds
            )
        except ImportError:
            logger.warning("XGBoost not available, skipping XGBoost training")
        
        # 最良モデルの選択
        self._select_best_model(results)
        
        # 結果の保存
        self._save_results(results, feature_names)
        
        # 可視化
        self._visualize_results(results)
        
        return results
    
    def _train_linear_regression(self, X_train, X_test, y_train, y_test, cv_folds):
        """線形回帰モデルの訓練"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        
        # 評価
        metrics = self._evaluate_model(y_test, y_pred)
        
        # クロスバリデーション
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=cv_folds, scoring='neg_mean_squared_error'
        )
        metrics['cv_rmse'] = np.sqrt(-cv_scores.mean())
        
        self.models['linear_regression'] = model
        return metrics
    
    def _train_ridge(self, X_train, X_test, y_train, y_test, cv_folds):
        """Ridge回帰モデルの訓練"""
        # ハイパーパラメータチューニング
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
        model = Ridge()
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        metrics = self._evaluate_model(y_test, y_pred)
        metrics['best_alpha'] = grid_search.best_params_['alpha']
        metrics['cv_rmse'] = np.sqrt(-grid_search.best_score_)
        
        self.models['ridge'] = best_model
        return metrics
    
    def _train_lasso(self, X_train, X_test, y_train, y_test, cv_folds):
        """Lasso回帰モデルの訓練"""
        # ハイパーパラメータチューニング
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        model = Lasso()
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        metrics = self._evaluate_model(y_test, y_pred)
        metrics['best_alpha'] = grid_search.best_params_['alpha']
        metrics['cv_rmse'] = np.sqrt(-grid_search.best_score_)
        
        self.models['lasso'] = best_model
        return metrics
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test, cv_folds):
        """Random Forestモデルの訓練"""
        # ハイパーパラメータチューニング
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(
            model, param_grid, cv=3,  # 計算時間削減のため3分割
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        metrics = self._evaluate_model(y_test, y_pred)
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_rmse'] = np.sqrt(-grid_search.best_score_)
        
        self.models['random_forest'] = best_model
        return metrics
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test, cv_folds):
        """XGBoostモデルの訓練"""
        import xgboost as xgb
        
        # ハイパーパラメータチューニング
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # 計算時間削減のため、簡易的なグリッドサーチ
        grid_search = GridSearchCV(
            model, param_grid, cv=3,
            scoring='neg_mean_squared_error', 
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        metrics = self._evaluate_model(y_test, y_pred)
        metrics['best_params'] = grid_search.best_params_
        metrics['cv_rmse'] = np.sqrt(-grid_search.best_score_)
        
        # 特徴量重要度
        feature_importance = best_model.feature_importances_
        metrics['feature_importance'] = dict(zip(
            self.preprocessor.feature_columns[:len(feature_importance)], 
            feature_importance.tolist()
        ))
        
        self.models['xgboost'] = best_model
        return metrics
    
    def _evaluate_model(self, y_true, y_pred):
        """モデルの評価指標を計算"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPEの計算（0除算を避ける）
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    def _select_best_model(self, results):
        """最良モデルを選択"""
        # R2スコアが最も高いモデルを選択
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.best_model = self.models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name}")
        logger.info(f"Best model R2 score: {results[best_model_name]['r2']:.4f}")
        
        # 最良モデルの保存
        best_model_path = self.model_dir / 'best_model.joblib'
        joblib.dump(self.best_model, best_model_path)
        logger.info(f"Best model saved to {best_model_path}")
        
        # モデルタイプの保存
        model_info = {
            'model_type': best_model_name,
            'metrics': results[best_model_name],
            'training_date': datetime.now().isoformat()
        }
        
        with open(self.model_dir / 'best_model_info.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    def _save_results(self, results, feature_names):
        """結果の保存"""
        # 全モデルの結果を保存
        results_path = self.model_dir / 'model_comparison_results.json'
        
        # NumPy配列をリストに変換
        serializable_results = {}
        for model_name, metrics in results.items():
            serializable_results[model_name] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_results[model_name][key] = float(value)
                else:
                    serializable_results[model_name][key] = value
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # 各モデルの保存
        for model_name, model in self.models.items():
            model_path = self.model_dir / f'{model_name}_model.joblib'
            joblib.dump(model, model_path)
            logger.info(f"{model_name} model saved to {model_path}")
    
    def _visualize_results(self, results):
        """結果の可視化"""
        # モデル比較のグラフ
        plt.figure(figsize=(12, 8))
        
        # 1. R2スコアの比較
        plt.subplot(2, 2, 1)
        models = list(results.keys())
        r2_scores = [results[m]['r2'] for m in models]
        plt.bar(models, r2_scores)
        plt.title('R² Score Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('R² Score')
        
        # 2. RMSEの比較
        plt.subplot(2, 2, 2)
        rmse_scores = [results[m]['rmse'] for m in models]
        plt.bar(models, rmse_scores)
        plt.title('RMSE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('RMSE (円)')
        
        # 3. MAEの比較
        plt.subplot(2, 2, 3)
        mae_scores = [results[m]['mae'] for m in models]
        plt.bar(models, mae_scores)
        plt.title('MAE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('MAE (円)')
        
        # 4. MAPEの比較
        plt.subplot(2, 2, 4)
        mape_scores = [results[m]['mape'] for m in models]
        plt.bar(models, mape_scores)
        plt.title('MAPE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('MAPE (%)')
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualization saved to model_comparison.png")
        
        # XGBoostの特徴量重要度を可視化
        if 'xgboost' in results and 'feature_importance' in results['xgboost']:
            plt.figure(figsize=(10, 8))
            importance = results['xgboost']['feature_importance']
            features = list(importance.keys())
            values = list(importance.values())
            
            # 上位20個の特徴量のみ表示
            indices = np.argsort(values)[-20:]
            plt.barh([features[i] for i in indices], [values[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('XGBoost Feature Importance (Top 20)')
            plt.tight_layout()
            plt.savefig(self.model_dir / 'xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def load_best_model(self):
        """最良モデルをロード"""
        best_model_path = self.model_dir / 'best_model.joblib'
        if best_model_path.exists():
            self.best_model = joblib.load(best_model_path)
            
            # モデル情報もロード
            info_path = self.model_dir / 'best_model_info.json'
            if info_path.exists():
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
            
            logger.info("Best model loaded successfully")
            return True
        return False
    
    def predict(self, 
                prefecture: str,
                city: str,
                district: str,
                land_area: float,
                building_area: float,
                building_age: int) -> Dict:
        """
        最良モデルで予測
        """
        if self.best_model is None:
            if not self.load_best_model():
                raise ValueError("No trained model available")
        
        # 前処理器から直接特徴量を作成（推論用）
        X = self.preprocessor.create_model_features(
            prefecture=prefecture,
            city=city,
            district=district,
            land_area=land_area,
            building_area=building_area,
            building_age=building_age
        )
        
        # 予測
        predicted_price = self.best_model.predict(X)[0]
        
        # 信頼区間の計算（簡易的な実装）
        confidence = 90.0 if self.model_info.get('model_type') == 'xgboost' else 85.0
        
        return {
            'estimated_price': float(predicted_price),
            'confidence': confidence,
            'price_range': {
                'min': float(predicted_price * 0.9),
                'max': float(predicted_price * 1.1)
            },
            'model_type': self.model_info.get('model_type', 'unknown'),
            'model_metrics': self.model_info.get('metrics', {})
        }


if __name__ == "__main__":
    # 使用例
    trainer = MultiModelTrainer()
    
    # 全モデルの訓練と比較
    logger.info("Starting multi-model training and comparison...")
    results = trainer.train_all_models(
        data_path="data/tokyo23_real_estate.csv",
        test_size=0.2,
        cv_folds=5
    )
    
    # 結果の表示
    print("\n=== Model Comparison Results ===")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  R² Score: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:,.0f} 円")
        print(f"  MAE: {metrics['mae']:,.0f} 円")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        
    # 予測テスト
    print("\n=== Prediction Test ===")
    try:
        test_prediction = trainer.predict(
            prefecture="東京都",
            city="港区",
            district="六本木",
            land_area=150,
            building_area=120,
            building_age=5
        )
        print(f"Predicted price: {test_prediction['estimated_price']:,.0f} 円")
        print(f"Model type: {test_prediction['model_type']}")
        print(f"Confidence: {test_prediction['confidence']}%")
    except Exception as e:
        print(f"Prediction test failed: {e}")
        print("モデルが正常に保存されました。個別テストは test_prediction.py を実行してください。")