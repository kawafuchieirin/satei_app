#!/usr/bin/env python3
"""
不動産査定モデル作成スクリプト

機能:
- データ収集・前処理
- 複数モデルタイプの学習・比較
- ハイパーパラメータ調整
- 評価・可視化
- モデル保存・デプロイ

使用方法:
python create_model.py [options]
"""

import argparse
import logging
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, validation_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
import joblib

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelCreator:
    """
    不動産査定モデル作成クラス
    """
    
    def __init__(self, output_dir='../api', random_state=42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        # 結果保存用
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        self.data = None
        self.feature_columns = [
            '都道府県', '市区町村', '地区', '土地面積', '建物面積', '築年数'
        ]
        
        # 可視化設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_and_prepare_data(self, data_source='sample', test_size=0.2):
        """
        データの読み込みと準備
        """
        logger.info(f"データ読み込み開始: {data_source}")
        
        if data_source == 'sample':
            self.data = self._generate_enhanced_sample_data()
        elif data_source == 'mlit':
            self.data = self._load_mlit_data()
        elif data_source == 'csv':
            self.data = self._load_csv_data()
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        logger.info(f"読み込み完了: {len(self.data)} 件")
        
        # データの前処理
        self.data = self._preprocess_data(self.data)
        logger.info(f"前処理後: {len(self.data)} 件")
        
        # 訓練・テスト分割
        self.X, self.y = self._prepare_features_target()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state
        )
        
        logger.info(f"データ分割完了 - 訓練: {len(self.X_train)}, テスト: {len(self.X_test)}")
        
        return self.data
    
    def _generate_enhanced_sample_data(self, n_samples=2000):
        """
        拡張サンプルデータの生成
        """
        np.random.seed(self.random_state)
        
        # 地域データの拡張
        prefectures = ['東京都', '神奈川県', '大阪府', '愛知県', '埼玉県', '千葉県', '兵庫県', '福岡県']
        cities = {
            '東京都': ['渋谷区', '新宿区', '港区', '千代田区', '中央区', '目黒区', '世田谷区'],
            '神奈川県': ['横浜市', '川崎市', '相模原市', '藤沢市', '茅ヶ崎市'],
            '大阪府': ['大阪市', '堺市', '東大阪市', '豊中市', '吹田市'],
            '愛知県': ['名古屋市', '豊田市', '岡崎市', '一宮市'],
            '埼玉県': ['さいたま市', '川口市', '所沢市', '越谷市'],
            '千葉県': ['千葉市', '船橋市', '柏市', '松戸市'],
            '兵庫県': ['神戸市', '姫路市', '西宮市', '尼崎市'],
            '福岡県': ['福岡市', '北九州市', '久留米市']
        }
        
        districts = ['中央', '東', '西', '南', '北', '駅前', '新町', '本町', '栄', '中心部']
        
        data = []
        
        for _ in range(n_samples):
            prefecture = np.random.choice(prefectures)
            city = np.random.choice(cities[prefecture])
            district = np.random.choice(districts)
            
            # 地域による価格係数
            region_multiplier = {
                '東京都': np.random.uniform(1.5, 3.0),
                '神奈川県': np.random.uniform(1.2, 2.0),
                '大阪府': np.random.uniform(1.1, 1.8),
                '愛知県': np.random.uniform(1.0, 1.5),
                '埼玉県': np.random.uniform(0.9, 1.4),
                '千葉県': np.random.uniform(0.8, 1.3),
                '兵庫県': np.random.uniform(0.9, 1.4),
                '福岡県': np.random.uniform(0.7, 1.2)
            }[prefecture]
            
            # 面積の生成（対数正規分布）
            land_area = np.random.lognormal(4.5, 0.6)  # 平均約100㎡
            building_area = land_area * np.random.uniform(0.6, 1.2)  # 建ぺい率考慮
            
            # 築年数（指数分布で新しい物件が多くなるように）
            building_age = np.random.exponential(15)
            building_age = min(building_age, 50)  # 最大50年
            
            # 価格計算（より現実的な計算式）
            base_price_per_sqm = 300000  # 基準価格（㎡あたり）
            
            # 各要因の影響
            area_factor = (land_area * 0.7 + building_area * 0.3)
            location_factor = region_multiplier
            age_factor = max(0.3, 1.0 - building_age * 0.015)  # 築年劣化
            
            # ランダム要因（市場変動など）
            random_factor = np.random.uniform(0.8, 1.2)
            
            price = (base_price_per_sqm * area_factor * location_factor * 
                    age_factor * random_factor)
            
            # 外れ値の追加（5%の確率）
            if np.random.random() < 0.05:
                price *= np.random.uniform(1.5, 3.0)
            
            data.append({
                '都道府県': prefecture,
                '市区町村': city,
                '地区': district,
                '土地面積': max(20, land_area),
                '建物面積': max(15, building_area),
                '築年数': int(building_age),
                '取引価格': int(price)
            })
        
        return pd.DataFrame(data)
    
    def _load_mlit_data(self):
        """
        MLIT APIからデータを読み込み
        """
        try:
            sys.path.append(str(Path(__file__).parent.parent / 'api'))
            from models.data_fetcher import MLITDataFetcher
            
            fetcher = MLITDataFetcher()
            df = fetcher.fetch_trade_data(
                prefecture="東京都",
                from_year=2021,
                to_year=2024
            )
            
            if df.empty:
                logger.warning("MLIT APIからデータを取得できませんでした。サンプルデータを使用します。")
                return self._generate_enhanced_sample_data()
            
            return df
            
        except Exception as e:
            logger.error(f"MLIT API エラー: {e}")
            return self._generate_enhanced_sample_data()
    
    def _load_csv_data(self):
        """
        CSVファイルからデータを読み込み
        """
        csv_path = self.output_dir / 'training_data.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            logger.warning(f"CSVファイルが見つかりません: {csv_path}")
            return self._generate_enhanced_sample_data()
    
    def _preprocess_data(self, df):
        """
        データの前処理
        """
        logger.info("データ前処理開始")
        
        df_clean = df.copy()
        original_count = len(df_clean)
        
        # 列名の統一
        column_mapping = {
            '取引価格（総額）': '取引価格',
            '面積（㎡）': '土地面積'
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        # 必要な列の確認
        required_columns = ['都道府県', '市区町村', '地区', '土地面積', '建物面積', '築年数', '取引価格']
        for col in required_columns:
            if col not in df_clean.columns:
                logger.warning(f"列が見つかりません: {col}")
        
        # 数値型への変換
        numeric_columns = ['土地面積', '建物面積', '築年数', '取引価格']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 異常値の除去
        df_clean = df_clean[
            (df_clean['取引価格'] > 1_000_000) &
            (df_clean['取引価格'] < 5_000_000_000) &
            (df_clean['土地面積'] > 10) &
            (df_clean['土地面積'] < 5000) &
            (df_clean['建物面積'] > 10) &
            (df_clean['建物面積'] < 2000) &
            (df_clean['築年数'] >= 0) &
            (df_clean['築年数'] <= 100)
        ]
        
        # 欠損値の除去
        df_clean = df_clean.dropna()
        
        removed_count = original_count - len(df_clean)
        logger.info(f"前処理完了: {removed_count} 件削除 ({removed_count/original_count*100:.1f}%)")
        
        return df_clean
    
    def _prepare_features_target(self):
        """
        特徴量とターゲットの準備
        """
        X = self.data[self.feature_columns].copy()
        y = self.data['取引価格']
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['都道府県', '市区町村', '地区']
        self.encoders = {}
        
        for col in categorical_columns:
            if col in X.columns:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder
        
        return X, y
    
    def create_model_configurations(self):
        """
        モデル設定の作成
        """
        configurations = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            
            'ridge': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Ridge(random_state=self.random_state))
                ]),
                'params': {
                    'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            
            'lasso': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Lasso(random_state=self.random_state))
                ]),
                'params': {
                    'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            
            'elastic_net': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', ElasticNet(random_state=self.random_state))
                ]),
                'params': {
                    'regressor__alpha': [0.1, 1.0, 10.0],
                    'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            }
        }
        
        return configurations
    
    def train_and_evaluate_models(self, use_grid_search=True, cv_folds=5):
        """
        モデルの学習と評価
        """
        logger.info("モデル学習・評価開始")
        
        configurations = self.create_model_configurations()
        
        for model_name, config in configurations.items():
            logger.info(f"学習中: {model_name}")
            
            try:
                if use_grid_search:
                    # ハイパーパラメータ調整
                    grid_search = GridSearchCV(
                        config['model'], 
                        config['params'],
                        cv=cv_folds,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(self.X_train, self.y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    # デフォルトパラメータで学習
                    best_model = config['model']
                    best_model.fit(self.X_train, self.y_train)
                    best_params = {}
                
                # 予測
                y_train_pred = best_model.predict(self.X_train)
                y_test_pred = best_model.predict(self.X_test)
                
                # 評価指標の計算
                metrics = {
                    'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                    'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                    'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                    'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                    'train_r2': r2_score(self.y_train, y_train_pred),
                    'test_r2': r2_score(self.y_test, y_test_pred),
                    'test_mape': mean_absolute_percentage_error(self.y_test, y_test_pred) * 100
                }
                
                # 交差検証
                cv_scores = cross_val_score(best_model, self.X, self.y, cv=cv_folds, scoring='r2')
                
                # 結果保存
                self.results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'metrics': metrics,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': {
                        'train': y_train_pred,
                        'test': y_test_pred
                    }
                }
                
                # ベストモデルの更新
                if metrics['test_r2'] > self.best_score:
                    self.best_score = metrics['test_r2']
                    self.best_model = best_model
                    self.best_model_name = model_name
                
                logger.info(f"{model_name} 完了 - Test R²: {metrics['test_r2']:.3f}")
                
            except Exception as e:
                logger.error(f"{model_name} でエラー: {e}")
                continue
        
        logger.info(f"最優秀モデル: {self.best_model_name} (R² = {self.best_score:.3f})")
    
    def create_visualizations(self):
        """
        可視化の作成
        """
        logger.info("可視化作成中")
        
        # 結果比較用のDataFrame作成
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Test R²': metrics['test_r2'],
                'Test MAE': metrics['test_mae'],
                'Test RMSE': metrics['test_rmse'],
                'Test MAPE': metrics['test_mape'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 1. モデル比較グラフ
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R²スコア比較
        axes[0, 0].bar(comparison_df['Model'], comparison_df['Test R²'])
        axes[0, 0].set_title('Test R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE比較
        axes[0, 1].bar(comparison_df['Model'], comparison_df['Test MAE'])
        axes[0, 1].set_title('Test MAE Comparison')
        axes[0, 1].set_ylabel('MAE (円)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 交差検証スコア
        axes[1, 0].errorbar(comparison_df['Model'], comparison_df['CV Mean'], 
                           yerr=comparison_df['CV Std'], fmt='o', capsize=5)
        axes[1, 0].set_title('Cross-Validation R² Score')
        axes[1, 0].set_ylabel('CV R² Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE比較
        axes[1, 1].bar(comparison_df['Model'], comparison_df['Test MAPE'])
        axes[1, 1].set_title('Test MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ベストモデルの予測精度
        best_result = self.results[self.best_model_name]
        y_test_pred = best_result['predictions']['test']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 実測値 vs 予測値
        axes[0].scatter(self.y_test, y_test_pred, alpha=0.6)
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('実測値')
        axes[0].set_ylabel('予測値')
        axes[0].set_title(f'{self.best_model_name} - 実測値 vs 予測値')
        
        # 残差プロット
        residuals = self.y_test - y_test_pred
        axes[1].scatter(y_test_pred, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('予測値')
        axes[1].set_ylabel('残差')
        axes[1].set_title(f'{self.best_model_name} - 残差プロット')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'best_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 特徴量重要度（該当するモデルの場合）
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = self.best_model.feature_importances_
            feature_names = self.feature_columns
            
            plt.figure(figsize=(10, 6))
            indices = np.argsort(feature_importance)[::-1]
            plt.bar(range(len(feature_importance)), feature_importance[indices])
            plt.xticks(range(len(feature_importance)), 
                      [feature_names[i] for i in indices], rotation=45)
            plt.title(f'{self.best_model_name} - 特徴量重要度')
            plt.ylabel('重要度')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("可視化保存完了")
    
    def save_results(self):
        """
        結果の保存
        """
        logger.info("結果保存中")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. ベストモデルの保存
        model_path = self.output_dir / 'valuation_model.joblib'
        joblib.dump(self.best_model, model_path)
        
        # 2. エンコーダーの保存
        encoders_path = self.output_dir / 'label_encoders.joblib'
        joblib.dump(self.encoders, encoders_path)
        
        # 3. 詳細結果の保存
        results_summary = {}
        for model_name, result in self.results.items():
            results_summary[model_name] = {
                'best_params': result['best_params'],
                'metrics': result['metrics'],
                'cv_mean': float(result['cv_mean']),
                'cv_std': float(result['cv_std'])
            }
        
        # 4. 学習情報の保存
        training_info = {
            'timestamp': timestamp,
            'best_model': self.best_model_name,
            'data_size': len(self.data),
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'feature_columns': self.feature_columns,
            'results': results_summary
        }
        
        info_path = self.output_dir / f'training_info_{timestamp}.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        # 5. 比較レポートの作成
        self._create_report(timestamp)
        
        logger.info(f"保存完了:")
        logger.info(f"  モデル: {model_path}")
        logger.info(f"  エンコーダー: {encoders_path}")
        logger.info(f"  学習情報: {info_path}")
    
    def _create_report(self, timestamp):
        """
        詳細レポートの作成
        """
        report_path = self.output_dir / f'model_report_{timestamp}.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 不動産査定モデル学習レポート\n\n")
            f.write(f"作成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
            
            f.write(f"## 概要\n\n")
            f.write(f"- **ベストモデル**: {self.best_model_name}\n")
            f.write(f"- **データサイズ**: {len(self.data):,} 件\n")
            f.write(f"- **訓練データ**: {len(self.X_train):,} 件\n")
            f.write(f"- **テストデータ**: {len(self.X_test):,} 件\n")
            f.write(f"- **最高スコア**: R² = {self.best_score:.3f}\n\n")
            
            f.write(f"## モデル比較結果\n\n")
            f.write(f"| モデル | Test R² | Test MAE | Test MAPE | CV Mean±Std |\n")
            f.write(f"|--------|---------|----------|-----------|-------------|\n")
            
            for model_name, result in self.results.items():
                metrics = result['metrics']
                f.write(f"| {model_name} | {metrics['test_r2']:.3f} | "
                       f"¥{metrics['test_mae']:,.0f} | {metrics['test_mape']:.1f}% | "
                       f"{result['cv_mean']:.3f}±{result['cv_std']:.3f} |\n")
            
            f.write(f"\n## ベストモデル詳細\n\n")
            best_result = self.results[self.best_model_name]
            f.write(f"**モデル**: {self.best_model_name}\n\n")
            f.write(f"**パラメータ**:\n")
            for param, value in best_result['best_params'].items():
                f.write(f"- {param}: {value}\n")
            
            f.write(f"\n**評価指標**:\n")
            metrics = best_result['metrics']
            f.write(f"- Test R²: {metrics['test_r2']:.3f}\n")
            f.write(f"- Test MAE: ¥{metrics['test_mae']:,.0f}\n")
            f.write(f"- Test RMSE: ¥{metrics['test_rmse']:,.0f}\n")
            f.write(f"- Test MAPE: {metrics['test_mape']:.1f}%\n")
            
            f.write(f"\n## 推奨事項\n\n")
            if self.best_score >= 0.8:
                f.write(f"✅ **優秀**: モデルの精度は非常に高く、本番環境での使用に適しています。\n")
            elif self.best_score >= 0.6:
                f.write(f"👍 **良好**: モデルの精度は良好です。継続的な改善により更なる向上が期待できます。\n")
            elif self.best_score >= 0.4:
                f.write(f"⚠️ **注意**: モデルの精度に改善の余地があります。特徴量の追加やハイパーパラメータ調整を推奨します。\n")
            else:
                f.write(f"🔴 **要改善**: モデルの精度が低く、データの見直しやアプローチの変更が必要です。\n")
        
        logger.info(f"レポート作成完了: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='不動産査定モデル作成スクリプト')
    parser.add_argument('--data-source', choices=['sample', 'mlit', 'csv'], 
                       default='sample', help='データソース')
    parser.add_argument('--output-dir', default='../api', 
                       help='出力ディレクトリ')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='ハイパーパラメータ調整をスキップ')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='交差検証の分割数')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='テストデータの割合')
    parser.add_argument('--random-state', type=int, default=42,
                       help='乱数シード')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏠 不動産査定モデル作成スクリプト")
    print("=" * 60)
    print(f"データソース: {args.data_source}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"ハイパーパラメータ調整: {'無効' if args.no_grid_search else '有効'}")
    print(f"交差検証分割数: {args.cv_folds}")
    print()
    
    try:
        # モデル作成器の初期化
        creator = ModelCreator(
            output_dir=args.output_dir,
            random_state=args.random_state
        )
        
        # データ読み込み・準備
        creator.load_and_prepare_data(
            data_source=args.data_source,
            test_size=args.test_size
        )
        
        # モデル学習・評価
        creator.train_and_evaluate_models(
            use_grid_search=not args.no_grid_search,
            cv_folds=args.cv_folds
        )
        
        # 可視化作成
        creator.create_visualizations()
        
        # 結果保存
        creator.save_results()
        
        print("\n" + "=" * 60)
        print("🎉 モデル作成完了！")
        print("=" * 60)
        print(f"ベストモデル: {creator.best_model_name}")
        print(f"最高スコア: R² = {creator.best_score:.3f}")
        print(f"出力先: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()