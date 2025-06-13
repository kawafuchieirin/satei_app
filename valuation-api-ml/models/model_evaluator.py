import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
from typing import Dict, List, Tuple, Optional
import warnings

from .data_fetcher import MLITDataFetcher
from .valuation_model import ValuationModel

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    不動産査定モデルの精度検証クラス
    """
    
    def __init__(self):
        self.model = None
        self.data = None
        self.encoders = {}
        self.feature_columns = [
            '都道府県', '市区町村', '地区', '土地面積', '建物面積', '築年数'
        ]
    
    def load_model_and_data(self):
        """
        モデルとデータの読み込み
        """
        logger.info("Loading model and data for evaluation...")
        
        # データ取得
        data_fetcher = MLITDataFetcher()
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
        self.data = self._preprocess_data(df)
        
        # モデル作成（既存のモデルがあれば使用、なければ新規作成）
        self.model = ValuationModel()
        
        if not self.data.empty:
            logger.info(f"Loaded {len(self.data)} records for evaluation")
        else:
            raise ValueError("No valid data available for evaluation")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データの前処理（ValuationModelと同じ処理）
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # 列名の統一
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
                (df_clean['取引価格'] > 1000000) &
                (df_clean['取引価格'] < 1000000000)
            ]
        
        if '土地面積' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['土地面積'] > 10) &
                (df_clean['土地面積'] < 1000)
            ]
        
        if '建物面積' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['建物面積'] > 10) &
                (df_clean['建物面積'] < 500)
            ]
        
        # 欠損値の除去
        df_clean = df_clean.dropna(subset=['取引価格'])
        
        return df_clean
    
    def evaluate_model_performance(self) -> Dict:
        """
        モデルの性能評価
        """
        if self.model is None or self.data is None:
            self.load_model_and_data()
        
        logger.info("Starting model performance evaluation...")
        
        # 特徴量とターゲットの準備
        X = self.data[self.feature_columns[:-1]].copy()
        X['築年数'] = self.data['築年数']
        y = self.data['取引価格']
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['都道府県', '市区町村', '地区']
        for col in categorical_columns:
            if col in X.columns:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder
        
        # 予測値の計算
        y_pred = self.model.model.predict(X)
        
        # 評価指標の計算
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        
        # 価格帯別の精度
        price_ranges = self._analyze_accuracy_by_price_range(y, y_pred)
        
        # 地域別の精度
        regional_accuracy = self._analyze_accuracy_by_region(X, y, y_pred)
        
        # 特徴量重要度
        feature_importance = self._get_feature_importance()
        
        results = {
            'overall_metrics': {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'mape': float(mape)
            },
            'price_range_accuracy': price_ranges,
            'regional_accuracy': regional_accuracy,
            'feature_importance': feature_importance,
            'sample_count': len(self.data),
            'evaluation_summary': self._generate_evaluation_summary(mae, r2, mape)
        }
        
        logger.info(f"Evaluation completed. R² Score: {r2:.3f}, MAE: {mae:.0f}")
        return results
    
    def cross_validate_model(self, cv_folds: int = 5) -> Dict:
        """
        交差検証による性能評価
        """
        if self.model is None or self.data is None:
            self.load_model_and_data()
        
        logger.info(f"Starting {cv_folds}-fold cross validation...")
        
        # 特徴量とターゲットの準備
        X = self.data[self.feature_columns[:-1]].copy()
        X['築年数'] = self.data['築年数']
        y = self.data['取引価格']
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['都道府県', '市区町村', '地区']
        for col in categorical_columns:
            if col in X.columns:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
        
        # 交差検証スコア
        cv_scores_r2 = cross_val_score(
            self.model.model, X, y, cv=cv_folds, scoring='r2'
        )
        cv_scores_neg_mae = cross_val_score(
            self.model.model, X, y, cv=cv_folds, scoring='neg_mean_absolute_error'
        )
        
        cv_results = {
            'r2_scores': cv_scores_r2.tolist(),
            'mae_scores': (-cv_scores_neg_mae).tolist(),
            'r2_mean': float(cv_scores_r2.mean()),
            'r2_std': float(cv_scores_r2.std()),
            'mae_mean': float((-cv_scores_neg_mae).mean()),
            'mae_std': float((-cv_scores_neg_mae).std()),
            'cv_folds': cv_folds
        }
        
        logger.info(f"Cross validation completed. Mean R² Score: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
        return cv_results
    
    def _analyze_accuracy_by_price_range(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        価格帯別の精度分析
        """
        df_analysis = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred
        })
        
        # 価格帯の定義
        price_ranges = [
            (0, 30_000_000, '低価格帯 (~3,000万円)'),
            (30_000_000, 70_000_000, '中価格帯 (3,000~7,000万円)'),
            (70_000_000, 150_000_000, '高価格帯 (7,000万円~1.5億円)'),
            (150_000_000, float('inf'), '超高価格帯 (1.5億円~)')
        ]
        
        results = {}
        for min_price, max_price, label in price_ranges:
            mask = (df_analysis['actual'] >= min_price) & (df_analysis['actual'] < max_price)
            subset = df_analysis[mask]
            
            if len(subset) > 0:
                mae = mean_absolute_error(subset['actual'], subset['predicted'])
                r2 = r2_score(subset['actual'], subset['predicted'])
                mape = mean_absolute_percentage_error(subset['actual'], subset['predicted']) * 100
                
                results[label] = {
                    'count': len(subset),
                    'mae': float(mae),
                    'r2_score': float(r2),
                    'mape': float(mape)
                }
        
        return results
    
    def _analyze_accuracy_by_region(self, X: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        地域別の精度分析
        """
        if '都道府県' not in X.columns:
            return {}
        
        df_analysis = pd.DataFrame({
            'prefecture': X['都道府県'],
            'actual': y_true,
            'predicted': y_pred
        })
        
        # 都道府県コードの逆変換
        prefecture_mapping = {v: k for k, v in self.encoders.get('都道府県', {}).items()} if self.encoders else {}
        
        results = {}
        for prefecture_code in df_analysis['prefecture'].unique():
            mask = df_analysis['prefecture'] == prefecture_code
            subset = df_analysis[mask]
            
            if len(subset) > 5:  # 十分なサンプル数がある場合のみ
                mae = mean_absolute_error(subset['actual'], subset['predicted'])
                r2 = r2_score(subset['actual'], subset['predicted'])
                
                prefecture_name = prefecture_mapping.get(prefecture_code, f'Prefecture_{prefecture_code}')
                results[prefecture_name] = {
                    'count': len(subset),
                    'mae': float(mae),
                    'r2_score': float(r2)
                }
        
        return results
    
    def _get_feature_importance(self) -> Dict:
        """
        特徴量重要度の取得
        """
        if hasattr(self.model.model, 'feature_importances_'):
            importance = self.model.model.feature_importances_
            feature_names = self.feature_columns
            
            # 重要度の高い順にソート
            importance_pairs = list(zip(feature_names, importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'features': [pair[0] for pair in importance_pairs],
                'importance': [float(pair[1]) for pair in importance_pairs]
            }
        else:
            return {'features': [], 'importance': []}
    
    def _generate_evaluation_summary(self, mae: float, r2: float, mape: float) -> List[str]:
        """
        評価結果のサマリー生成
        """
        summary = []
        
        # R²スコアの評価
        if r2 >= 0.8:
            summary.append("R²スコアが0.8以上で、非常に高い精度です")
        elif r2 >= 0.6:
            summary.append("R²スコアが0.6以上で、良好な精度です")
        elif r2 >= 0.4:
            summary.append("R²スコアが0.4以上で、改善の余地があります")
        else:
            summary.append("R²スコアが0.4未満で、モデルの見直しが必要です")
        
        # MAPEの評価
        if mape <= 10:
            summary.append("MAPE（平均絶対パーセント誤差）が10%以下で、高精度です")
        elif mape <= 20:
            summary.append("MAPE（平均絶対パーセント誤差）が20%以下で、実用的な精度です")
        elif mape <= 30:
            summary.append("MAPE（平均絶対パーセント誤差）が30%以下で、参考程度の精度です")
        else:
            summary.append("MAPE（平均絶対パーセント誤差）が30%を超えており、精度が不十分です")
        
        # MAEの評価（価格の観点から）
        if mae <= 5_000_000:
            summary.append("平均絶対誤差が500万円以下で、実用的な精度です")
        elif mae <= 10_000_000:
            summary.append("平均絶対誤差が1,000万円以下で、参考になる精度です")
        else:
            summary.append("平均絶対誤差が1,000万円を超えており、精度の改善が必要です")
        
        return summary
    
    def generate_prediction_samples(self, n_samples: int = 10) -> List[Dict]:
        """
        予測サンプルの生成（デモ用）
        """
        if self.data is None:
            self.load_model_and_data()
        
        # ランダムサンプルを選択
        sample_data = self.data.sample(n=min(n_samples, len(self.data)))
        
        samples = []
        for _, row in sample_data.iterrows():
            try:
                # モデルで予測
                prediction = self.model.predict(
                    prefecture=row.get('都道府県', '東京都'),
                    city=row.get('市区町村', '渋谷区'),
                    district=row.get('地区', '恵比寿'),
                    land_area=row['土地面積'],
                    building_area=row['建物面積'],
                    building_age=row['築年数']
                )
                
                actual_price = row['取引価格']
                predicted_price = prediction['estimated_price']
                error_rate = abs(predicted_price - actual_price) / actual_price * 100
                
                samples.append({
                    'input': {
                        'prefecture': row.get('都道府県', '東京都'),
                        'city': row.get('市区町村', '渋谷区'),
                        'district': row.get('地区', '恵比寿'),
                        'land_area': row['土地面積'],
                        'building_area': row['建物面積'],
                        'building_age': row['築年数']
                    },
                    'actual_price': float(actual_price),
                    'predicted_price': float(predicted_price),
                    'error_rate': float(error_rate),
                    'absolute_error': float(abs(predicted_price - actual_price))
                })
            except Exception as e:
                logger.warning(f"Failed to generate prediction sample: {e}")
                continue
        
        return samples