#!/usr/bin/env python3
"""
バッチ処理用モデル学習スクリプト

複数の設定でモデルを学習し、最適なモデルを自動選択します。

使用方法:
python batch_model_training.py [options]
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import concurrent.futures
import multiprocessing

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchModelTrainer:
    """
    バッチ処理用モデル学習クラス
    """
    
    def __init__(self, output_dir='../api', random_state=42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.results = []
        self.best_config = None
        self.best_score = -np.inf
        
        # CPUコア数の取得
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1)
        
    def generate_model_configurations(self):
        """
        モデル設定のバッチ生成
        """
        configurations = []
        
        # Random Forest の設定バリエーション
        rf_configs = [
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
            {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2},
            {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 10},
            {'n_estimators': 150, 'max_depth': None, 'min_samples_split': 5},
        ]
        
        for config in rf_configs:
            configurations.append({
                'name': f"rf_{config['n_estimators']}_{config['max_depth']}_{config['min_samples_split']}",
                'model_type': 'random_forest',
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    **config
                ),
                'params': config
            })
        
        # Gradient Boosting の設定バリエーション
        gb_configs = [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3},
            {'n_estimators': 150, 'learning_rate': 0.15, 'max_depth': 7},
        ]
        
        for config in gb_configs:
            configurations.append({
                'name': f"gb_{config['n_estimators']}_{config['learning_rate']}_{config['max_depth']}",
                'model_type': 'gradient_boosting',
                'model': GradientBoostingRegressor(
                    random_state=self.random_state,
                    **config
                ),
                'params': config
            })
        
        # Ridge回帰の設定バリエーション
        ridge_alphas = [0.1, 1.0, 10.0, 100.0]
        for alpha in ridge_alphas:
            configurations.append({
                'name': f"ridge_{alpha}",
                'model_type': 'ridge',
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Ridge(alpha=alpha, random_state=self.random_state))
                ]),
                'params': {'alpha': alpha}
            })
        
        # Lasso回帰の設定バリエーション
        lasso_alphas = [0.1, 1.0, 10.0]
        for alpha in lasso_alphas:
            configurations.append({
                'name': f"lasso_{alpha}",
                'model_type': 'lasso',
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', Lasso(alpha=alpha, random_state=self.random_state))
                ]),
                'params': {'alpha': alpha}
            })
        
        return configurations
    
    def load_and_prepare_data(self, data_source='sample'):
        """
        データの読み込みと準備
        """
        logger.info(f"データ読み込み開始: {data_source}")
        
        if data_source == 'sample':
            self.data = self._generate_sample_data()
        elif data_source == 'mlit':
            self.data = self._load_mlit_data()
        elif data_source == 'csv':
            self.data = self._load_csv_data()
        
        # データ前処理
        self.data = self._preprocess_data(self.data)
        
        # 特徴量準備
        feature_columns = ['都道府県', '市区町村', '地区', '土地面積', '建物面積', '築年数']
        X = self.data[feature_columns].copy()
        y = self.data['取引価格']
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['都道府県', '市区町村', '地区']
        self.encoders = {}
        
        for col in categorical_columns:
            if col in X.columns:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder
        
        # 訓練・テスト分割
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        logger.info(f"データ準備完了 - 訓練: {len(self.X_train)}, テスト: {len(self.X_test)}")
    
    def _generate_sample_data(self, n_samples=2000):
        """
        サンプルデータ生成
        """
        np.random.seed(self.random_state)
        
        prefectures = ['東京都', '神奈川県', '大阪府', '愛知県', '埼玉県']
        cities = ['渋谷区', '新宿区', '港区', '横浜市', '川崎市', '大阪市', '名古屋市', 'さいたま市']
        districts = ['中央', '東', '西', '南', '北', '駅前']
        
        data = []
        for _ in range(n_samples):
            prefecture = np.random.choice(prefectures)
            city = np.random.choice(cities)
            district = np.random.choice(districts)
            
            land_area = np.random.lognormal(4.5, 0.6)
            building_area = land_area * np.random.uniform(0.6, 1.2)
            building_age = np.random.exponential(15)
            building_age = min(building_age, 50)
            
            # 価格計算
            base_price = 300000
            location_factor = np.random.uniform(0.8, 2.5)
            age_factor = max(0.3, 1.0 - building_age * 0.015)
            price = base_price * (land_area * 0.7 + building_area * 0.3) * location_factor * age_factor
            
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
            df = fetcher.fetch_trade_data(prefecture="東京都", from_year=2021, to_year=2024)
            
            if df.empty:
                logger.warning("MLIT APIからデータを取得できませんでした。サンプルデータを使用します。")
                return self._generate_sample_data()
            
            return df
        except Exception as e:
            logger.error(f"MLIT API エラー: {e}")
            return self._generate_sample_data()
    
    def _load_csv_data(self):
        """
        CSVファイルからデータを読み込み
        """
        csv_path = self.output_dir / 'training_data.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            logger.warning(f"CSVファイルが見つかりません: {csv_path}")
            return self._generate_sample_data()
    
    def _preprocess_data(self, df):
        """
        データ前処理
        """
        df_clean = df.copy()
        
        # 列名統一
        column_mapping = {
            '取引価格（総額）': '取引価格',
            '面積（㎡）': '土地面積'
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        # 数値変換
        numeric_columns = ['土地面積', '建物面積', '築年数', '取引価格']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 異常値除去
        df_clean = df_clean[
            (df_clean['取引価格'] > 1_000_000) &
            (df_clean['取引価格'] < 3_000_000_000) &
            (df_clean['土地面積'] > 10) &
            (df_clean['土地面積'] < 2000) &
            (df_clean['建物面積'] > 10) &
            (df_clean['建物面積'] < 1000) &
            (df_clean['築年数'] >= 0) &
            (df_clean['築年数'] <= 80)
        ]
        
        # 欠損値除去
        df_clean = df_clean.dropna()
        
        return df_clean
    
    def train_single_model(self, config):
        """
        単一モデルの学習（並列処理用）
        """
        try:
            start_time = time.time()
            
            # モデル学習
            model = config['model']
            model.fit(self.X_train, self.y_train)
            
            # 予測
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # 評価指標計算
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # 交差検証
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='r2')
            
            training_time = time.time() - start_time
            
            result = {
                'name': config['name'],
                'model_type': config['model_type'],
                'params': config['params'],
                'model': model,
                'metrics': {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                },
                'training_time': training_time
            }
            
            logger.info(f"{config['name']} 完了 - Test R²: {test_r2:.3f}, 時間: {training_time:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"{config['name']} でエラー: {e}")
            return None
    
    def run_batch_training(self, max_workers=None):
        """
        バッチ学習の実行
        """
        logger.info("バッチ学習開始")
        
        configurations = self.generate_model_configurations()
        logger.info(f"学習予定モデル数: {len(configurations)}")
        
        if max_workers is None:
            max_workers = min(self.n_jobs, len(configurations))
        
        # 並列学習実行
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 全ての設定を並列で実行
            future_to_config = {
                executor.submit(self.train_single_model, config): config 
                for config in configurations
            }
            
            # 結果収集
            for future in concurrent.futures.as_completed(future_to_config):
                result = future.result()
                if result is not None:
                    self.results.append(result)
                    
                    # ベストモデル更新
                    if result['metrics']['test_r2'] > self.best_score:
                        self.best_score = result['metrics']['test_r2']
                        self.best_config = result
        
        # 結果ソート
        self.results.sort(key=lambda x: x['metrics']['test_r2'], reverse=True)
        
        logger.info(f"バッチ学習完了 - ベストモデル: {self.best_config['name']} (R² = {self.best_score:.3f})")
    
    def save_best_model(self):
        """
        最優秀モデルの保存
        """
        if self.best_config is None:
            logger.error("ベストモデルが見つかりません")
            return
        
        logger.info("ベストモデル保存中")
        
        # モデル保存
        model_path = self.output_dir / 'valuation_model.joblib'
        joblib.dump(self.best_config['model'], model_path)
        
        # エンコーダー保存
        encoders_path = self.output_dir / 'label_encoders.joblib'
        joblib.dump(self.encoders, encoders_path)
        
        # 学習情報保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        training_info = {
            'timestamp': timestamp,
            'best_model': {
                'name': self.best_config['name'],
                'type': self.best_config['model_type'],
                'params': self.best_config['params'],
                'metrics': self.best_config['metrics'],
                'training_time': self.best_config['training_time']
            },
            'all_results': [
                {
                    'name': result['name'],
                    'type': result['model_type'],
                    'metrics': result['metrics'],
                    'training_time': result['training_time']
                }
                for result in self.results
            ],
            'data_info': {
                'total_samples': len(self.data),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test)
            }
        }
        
        info_path = self.output_dir / f'batch_training_results_{timestamp}.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存完了:")
        logger.info(f"  モデル: {model_path}")
        logger.info(f"  エンコーダー: {encoders_path}")
        logger.info(f"  結果: {info_path}")
    
    def print_results_summary(self):
        """
        結果サマリーの表示
        """
        print("\n" + "=" * 80)
        print("🏆 バッチ学習結果サマリー")
        print("=" * 80)
        
        print(f"学習済みモデル数: {len(self.results)}")
        print(f"ベストモデル: {self.best_config['name']}")
        print(f"最高スコア: R² = {self.best_score:.3f}")
        print()
        
        print("📊 上位5モデル:")
        print("-" * 80)
        print(f"{'順位':<4} {'モデル名':<20} {'Test R²':<10} {'Test MAE':<15} {'学習時間':<10}")
        print("-" * 80)
        
        for i, result in enumerate(self.results[:5], 1):
            metrics = result['metrics']
            print(f"{i:<4} {result['name']:<20} {metrics['test_r2']:<10.3f} "
                  f"¥{metrics['test_mae']:<13,.0f} {result['training_time']:<10.1f}s")
        
        print()
        
        # モデルタイプ別統計
        type_stats = {}
        for result in self.results:
            model_type = result['model_type']
            if model_type not in type_stats:
                type_stats[model_type] = []
            type_stats[model_type].append(result['metrics']['test_r2'])
        
        print("📈 モデルタイプ別統計:")
        print("-" * 50)
        for model_type, scores in type_stats.items():
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            print(f"{model_type:<20} 平均: {avg_score:.3f}, 最高: {max_score:.3f}")


def main():
    parser = argparse.ArgumentParser(description='バッチ処理用モデル学習スクリプト')
    parser.add_argument('--data-source', choices=['sample', 'mlit', 'csv'], 
                       default='sample', help='データソース')
    parser.add_argument('--output-dir', default='../api', 
                       help='出力ディレクトリ')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='並列処理の最大ワーカー数')
    parser.add_argument('--random-state', type=int, default=42,
                       help='乱数シード')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 バッチ処理モデル学習スクリプト")
    print("=" * 60)
    print(f"データソース: {args.data_source}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"最大ワーカー数: {args.max_workers or 'auto'}")
    print()
    
    try:
        # バッチ学習器の初期化
        trainer = BatchModelTrainer(
            output_dir=args.output_dir,
            random_state=args.random_state
        )
        
        # データ準備
        trainer.load_and_prepare_data(data_source=args.data_source)
        
        # バッチ学習実行
        trainer.run_batch_training(max_workers=args.max_workers)
        
        # 結果表示
        trainer.print_results_summary()
        
        # ベストモデル保存
        trainer.save_best_model()
        
        print("\n🎉 バッチ学習完了！")
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()