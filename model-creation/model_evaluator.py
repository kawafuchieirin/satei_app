#!/usr/bin/env python3
"""
モデル評価・比較スクリプト
複数の機械学習モデルの性能を評価し、最適なモデルを選択
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """モデル評価クラス"""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.evaluation_results = {}
        
    def evaluate_all_models(self, X_test, y_test, model_names: Optional[List[str]] = None):
        """
        全モデルの評価
        
        Args:
            X_test: テストデータの特徴量
            y_test: テストデータの目的変数
            model_names: 評価するモデル名のリスト
        """
        if model_names is None:
            model_names = ['linear_regression', 'ridge', 'lasso', 'random_forest', 'xgboost', 'lightgbm']
        
        for model_name in model_names:
            model_path = self.model_dir / f'{model_name}_model.joblib'
            if model_path.exists():
                logger.info(f"Evaluating {model_name}...")
                model = joblib.load(model_path)
                self.evaluation_results[model_name] = self._evaluate_single_model(
                    model, X_test, y_test, model_name
                )
        
        return self.evaluation_results
    
    def _evaluate_single_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """単一モデルの評価"""
        # 予測
        y_pred = model.predict(X_test)
        
        # 基本的な評価指標
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        # 価格帯別の精度
        price_ranges = [
            (0, 30000000, "〜3000万円"),
            (30000000, 50000000, "3000万〜5000万円"),
            (50000000, 80000000, "5000万〜8000万円"),
            (80000000, 100000000, "8000万〜1億円"),
            (100000000, float('inf'), "1億円〜")
        ]
        
        range_accuracy = {}
        for min_price, max_price, label in price_ranges:
            mask = (y_test >= min_price) & (y_test < max_price)
            if mask.sum() > 0:
                range_mae = mean_absolute_error(y_test[mask], y_pred[mask])
                range_mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
                range_accuracy[label] = {
                    'count': mask.sum(),
                    'mae': range_mae,
                    'mape': range_mape
                }
        
        # 予測の偏り分析
        residuals = y_test - y_pred
        bias = np.mean(residuals)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'bias': bias,
            'range_accuracy': range_accuracy,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist()
            }
        }
    
    def create_evaluation_report(self, output_path: Optional[str] = None):
        """評価レポートの作成"""
        if not self.evaluation_results:
            logger.error("No evaluation results available")
            return
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'models_evaluated': list(self.evaluation_results.keys()),
            'summary': {},
            'detailed_results': self.evaluation_results
        }
        
        # サマリーの作成
        for model_name, results in self.evaluation_results.items():
            report['summary'][model_name] = {
                'r2_score': results['r2'],
                'rmse': results['rmse'],
                'mae': results['mae'],
                'mape': results['mape']
            }
        
        # 最良モデルの特定
        best_model = max(self.evaluation_results.items(), 
                        key=lambda x: x[1]['r2'])[0]
        report['best_model'] = {
            'name': best_model,
            'r2_score': self.evaluation_results[best_model]['r2']
        }
        
        # レポートの保存
        if output_path is None:
            output_path = self.model_dir / 'evaluation_report.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return report
    
    def visualize_model_comparison(self):
        """モデル比較の可視化"""
        if not self.evaluation_results:
            logger.error("No evaluation results available")
            return
        
        # 図のセットアップ
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        models = list(self.evaluation_results.keys())
        
        # 1. R²スコア比較
        ax = axes[0, 0]
        r2_scores = [self.evaluation_results[m]['r2'] for m in models]
        ax.bar(models, r2_scores)
        ax.set_title('R² Score')
        ax.set_xticklabels(models, rotation=45)
        ax.set_ylim(0, 1)
        
        # 2. RMSE比較
        ax = axes[0, 1]
        rmse_scores = [self.evaluation_results[m]['rmse'] for m in models]
        ax.bar(models, rmse_scores)
        ax.set_title('RMSE (円)')
        ax.set_xticklabels(models, rotation=45)
        
        # 3. MAPE比較
        ax = axes[0, 2]
        mape_scores = [self.evaluation_results[m]['mape'] for m in models]
        ax.bar(models, mape_scores)
        ax.set_title('MAPE (%)')
        ax.set_xticklabels(models, rotation=45)
        
        # 4. 予測精度散布図（最良モデル）
        best_model = max(self.evaluation_results.items(), 
                        key=lambda x: x[1]['r2'])[0]
        ax = axes[1, 0]
        y_true = self.evaluation_results[best_model]['predictions']['y_true']
        y_pred = self.evaluation_results[best_model]['predictions']['y_pred']
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        ax.set_xlabel('Actual Price (円)')
        ax.set_ylabel('Predicted Price (円)')
        ax.set_title(f'Best Model ({best_model}) - Predictions')
        
        # 5. 価格帯別精度（最良モデル）
        ax = axes[1, 1]
        range_acc = self.evaluation_results[best_model]['range_accuracy']
        if range_acc:
            ranges = list(range_acc.keys())
            mapes = [range_acc[r]['mape'] for r in ranges]
            ax.bar(ranges, mapes)
            ax.set_title('MAPE by Price Range')
            ax.set_xlabel('Price Range')
            ax.set_ylabel('MAPE (%)')
            ax.set_xticklabels(ranges, rotation=45)
        
        # 6. モデル別バイアス
        ax = axes[1, 2]
        biases = [self.evaluation_results[m]['bias'] for m in models]
        ax.bar(models, biases)
        ax.set_title('Prediction Bias')
        ax.set_xticklabels(models, rotation=45)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_ylabel('Mean Residual (円)')
        
        plt.tight_layout()
        output_path = self.model_dir / 'model_comparison_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def create_detailed_analysis(self, model_name: str):
        """特定モデルの詳細分析"""
        if model_name not in self.evaluation_results:
            logger.error(f"No results available for {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        
        # 残差分析
        y_true = np.array(results['predictions']['y_true'])
        y_pred = np.array(results['predictions']['y_pred'])
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_name} - Detailed Analysis', fontsize=16)
        
        # 1. 残差プロット
        ax = axes[0, 0]
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Price')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        
        # 2. 残差ヒストグラム
        ax = axes[0, 1]
        ax.hist(residuals, bins=30, edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        
        # 3. Q-Qプロット
        ax = axes[1, 0]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        
        # 4. 予測誤差の価格依存性
        ax = axes[1, 1]
        relative_errors = np.abs(residuals) / y_true * 100
        ax.scatter(y_true, relative_errors, alpha=0.5)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Relative Error (%)')
        ax.set_title('Relative Error vs Price')
        
        plt.tight_layout()
        output_path = self.model_dir / f'{model_name}_detailed_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Detailed analysis saved to {output_path}")
        
        # 統計サマリー
        stats_summary = {
            'residual_mean': float(np.mean(residuals)),
            'residual_std': float(np.std(residuals)),
            'residual_skewness': float(stats.skew(residuals)),
            'residual_kurtosis': float(stats.kurtosis(residuals)),
            'relative_error_mean': float(np.mean(relative_errors)),
            'relative_error_std': float(np.std(relative_errors))
        }
        
        with open(self.model_dir / f'{model_name}_stats_summary.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        return stats_summary


def evaluate_models_from_file(data_path: str, model_dir: str = 'models'):
    """ファイルからデータを読み込んでモデルを評価"""
    from data_preprocessor import RealEstateDataPreprocessor
    from sklearn.model_selection import train_test_split
    
    # データの読み込み
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 前処理
    preprocessor = RealEstateDataPreprocessor()
    features_df = preprocessor.prepare_features(df)
    X, y, feature_names = preprocessor.preprocess(features_df, is_training=True)
    
    # データ分割
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 評価実行
    evaluator = ModelEvaluator(model_dir)
    results = evaluator.evaluate_all_models(X_test, y_test)
    
    # レポート作成
    report = evaluator.create_evaluation_report()
    evaluator.visualize_model_comparison()
    
    # 最良モデルの詳細分析
    if report and 'best_model' in report:
        evaluator.create_detailed_analysis(report['best_model']['name'])
    
    return results, report


if __name__ == "__main__":
    # 使用例
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # デフォルトのデータパスを探す
        data_files = list(Path('data').glob('tokyo23_real_estate_*.csv'))
        if data_files:
            data_path = str(data_files[-1])  # 最新のファイルを使用
        else:
            logger.error("No data file found. Please provide a data path.")
            sys.exit(1)
    
    logger.info("Starting model evaluation...")
    results, report = evaluate_models_from_file(data_path)
    
    print("\n=== Model Evaluation Summary ===")
    if report:
        for model_name, summary in report['summary'].items():
            print(f"\n{model_name}:")
            print(f"  R² Score: {summary['r2_score']:.4f}")
            print(f"  RMSE: {summary['rmse']:,.0f} 円")
            print(f"  MAE: {summary['mae']:,.0f} 円")
            print(f"  MAPE: {summary['mape']:.2f}%")
        
        print(f"\nBest Model: {report['best_model']['name']} (R² = {report['best_model']['r2_score']:.4f})")