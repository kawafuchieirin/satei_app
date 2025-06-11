#!/usr/bin/env python3
"""
不動産査定モデルの学習・作成スクリプト

使用方法:
python train_model.py [options]

オプション:
--data-source: データソース (mlit|sample|csv) デフォルト: mlit
--model-type: モデルタイプ (rf|gb|linear) デフォルト: rf  
--output-dir: モデル保存ディレクトリ デフォルト: ./
--evaluate: 学習後に評価を実行
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from models.data_fetcher import MLITDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    不動産査定モデルの学習クラス
    """
    
    def __init__(self, model_type='rf', output_dir='./'):
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.model = None
        self.encoders = {}
        self.scaler = None
        self.feature_columns = [
            '都道府県', '市区町村', '地区', '土地面積', '建物面積', '築年数'
        ]
        self.training_info = {}
    
    def load_data(self, data_source='mlit'):
        """
        データの読み込み
        
        Args:
            data_source: データソース ('mlit', 'sample', 'csv')
        """
        logger.info(f"Loading data from source: {data_source}")
        
        if data_source == 'mlit':
            df = self._load_mlit_data()
        elif data_source == 'sample':
            df = self._load_sample_data()
        elif data_source == 'csv':
            df = self._load_csv_data()
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        if df.empty:
            raise ValueError("No data loaded")
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def _load_mlit_data(self):
        """
        MLIT APIからデータを取得
        """
        data_fetcher = MLITDataFetcher()
        
        # 複数の都道府県から取得を試行
        prefectures = ['東京都', '神奈川県', '大阪府', '愛知県']
        all_data = []
        
        for prefecture in prefectures:
            try:
                logger.info(f"Fetching data for {prefecture}")
                df = data_fetcher.fetch_trade_data(
                    prefecture=prefecture,
                    from_year=2021,
                    to_year=2024
                )
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Got {len(df)} records from {prefecture}")
            except Exception as e:
                logger.warning(f"Failed to fetch data for {prefecture}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined {len(combined_df)} records from MLIT API")
            return combined_df
        else:
            logger.warning("No data from MLIT API, falling back to sample data")
            return self._load_sample_data()
    
    def _load_sample_data(self):
        """
        サンプルデータを生成
        """
        data_fetcher = MLITDataFetcher()
        return data_fetcher.generate_sample_data()
    
    def _load_csv_data(self):
        """
        CSVファイルからデータを読み込み
        """
        csv_path = self.output_dir / 'training_data.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            logger.warning(f"CSV file not found: {csv_path}")
            return pd.DataFrame()
    
    def preprocess_data(self, df):
        """
        データの前処理
        """
        logger.info("Preprocessing data...")
        
        df_clean = df.copy()
        
        # 列名の統一
        column_mapping = {
            '取引価格（総額）': '取引価格',
            '面積（㎡）': '土地面積'
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        # 必要な列の確認・追加
        required_columns = ['都道府県', '市区町村', '地区', '土地面積', '建物面積', '築年数', '取引価格']
        for col in required_columns:
            if col not in df_clean.columns:
                if col == '都道府県':
                    df_clean[col] = '東京都'
                elif col == '市区町村':
                    df_clean[col] = '渋谷区'
                elif col == '地区':
                    df_clean[col] = '恵比寿'
                else:
                    logger.warning(f"Missing required column: {col}")
        
        # 数値列の処理
        numeric_columns = ['土地面積', '建物面積', '築年数', '取引価格']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # 異常値の除去
        original_count = len(df_clean)
        
        # 価格の異常値除去
        if '取引価格' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['取引価格'] > 1_000_000) &  # 100万円以上
                (df_clean['取引価格'] < 2_000_000_000)  # 20億円未満
            ]
        
        # 面積の異常値除去
        if '土地面積' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['土地面積'] > 5) &  # 5㎡以上
                (df_clean['土地面積'] < 2000)  # 2000㎡未満
            ]
        
        if '建物面積' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['建物面積'] > 5) &  # 5㎡以上
                (df_clean['建物面積'] < 1000)  # 1000㎡未満
            ]
        
        # 築年数の異常値除去
        if '築年数' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['築年数'] >= 0) &  # 0年以上
                (df_clean['築年数'] <= 100)  # 100年以下
            ]
        
        # 欠損値の除去
        df_clean = df_clean.dropna(subset=['取引価格'])
        
        removed_count = original_count - len(df_clean)
        logger.info(f"Removed {removed_count} outlier records ({removed_count/original_count*100:.1f}%)")
        
        return df_clean
    
    def create_model(self):
        """
        モデルの作成
        """
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Created {self.model_type} model: {self.model}")
    
    def train_model(self, df):
        """
        モデルの学習
        """
        logger.info("Starting model training...")
        
        # 特徴量とターゲットの分離
        X = df[self.feature_columns].copy()
        y = df['取引価格']
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['都道府県', '市区町村', '地区']
        for col in categorical_columns:
            if col in X.columns:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder
        
        # 数値列の標準化（線形回帰の場合）
        if self.model_type == 'linear':
            self.scaler = StandardScaler()
            numeric_columns = ['土地面積', '建物面積', '築年数']
            X[numeric_columns] = self.scaler.fit_transform(X[numeric_columns])
        
        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # モデル学習
        logger.info(f"Training with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # 予測と評価
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 評価指標の計算
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # 交差検証
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
        
        # 学習情報の保存
        self.training_info = {
            'model_type': self.model_type,
            'training_date': datetime.now().isoformat(),
            'data_size': len(df),
            'feature_columns': self.feature_columns,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'metrics': {
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'cv_r2_mean': float(cv_scores.mean()),
                'cv_r2_std': float(cv_scores.std())
            }
        }
        
        logger.info(f"Training completed:")
        logger.info(f"  Test MAE: ¥{test_mae:,.0f}")
        logger.info(f"  Test R²: {test_r2:.3f}")
        logger.info(f"  CV R² (mean±std): {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        return X_test, y_test, y_test_pred
    
    def save_model(self):
        """
        モデルの保存
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # モデルファイルの保存
        model_path = self.output_dir / f'valuation_model_{self.model_type}_{timestamp}.joblib'
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # エンコーダーの保存
        if self.encoders:
            encoders_path = self.output_dir / f'label_encoders_{self.model_type}_{timestamp}.joblib'
            joblib.dump(self.encoders, encoders_path)
            logger.info(f"Encoders saved to: {encoders_path}")
        
        # スケーラーの保存
        if self.scaler:
            scaler_path = self.output_dir / f'scaler_{self.model_type}_{timestamp}.joblib'
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
        
        # 学習情報の保存
        info_path = self.output_dir / f'training_info_{self.model_type}_{timestamp}.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_info, f, ensure_ascii=False, indent=2)
        logger.info(f"Training info saved to: {info_path}")
        
        # 現在のモデルとしてコピー（本番用）
        current_model_path = self.output_dir / 'valuation_model.joblib'
        current_encoders_path = self.output_dir / 'label_encoders.joblib'
        
        joblib.dump(self.model, current_model_path)
        if self.encoders:
            joblib.dump(self.encoders, current_encoders_path)
        if self.scaler:
            current_scaler_path = self.output_dir / 'scaler.joblib'
            joblib.dump(self.scaler, current_scaler_path)
        
        logger.info("Current model files updated for production use")
        
        return {
            'model_path': str(model_path),
            'encoders_path': str(encoders_path) if self.encoders else None,
            'scaler_path': str(scaler_path) if self.scaler else None,
            'info_path': str(info_path)
        }


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description='Train real estate valuation model')
    parser.add_argument('--data-source', choices=['mlit', 'sample', 'csv'], 
                       default='mlit', help='Data source')
    parser.add_argument('--model-type', choices=['rf', 'gb', 'linear'], 
                       default='rf', help='Model type')
    parser.add_argument('--output-dir', default='./', 
                       help='Output directory for model files')
    parser.add_argument('--evaluate', action='store_true', 
                       help='Run evaluation after training')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("不動産査定モデル 学習スクリプト")
    print("=" * 60)
    print(f"データソース: {args.data_source}")
    print(f"モデルタイプ: {args.model_type}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print()
    
    try:
        # モデル学習器の初期化
        trainer = ModelTrainer(
            model_type=args.model_type,
            output_dir=args.output_dir
        )
        
        # データ読み込み
        df = trainer.load_data(args.data_source)
        
        # データ前処理
        df_processed = trainer.preprocess_data(df)
        
        # モデル作成
        trainer.create_model()
        
        # モデル学習
        X_test, y_test, y_pred = trainer.train_model(df_processed)
        
        # モデル保存
        saved_files = trainer.save_model()
        
        print("\n" + "=" * 60)
        print("学習完了！")
        print("=" * 60)
        print("保存されたファイル:")
        for key, path in saved_files.items():
            if path:
                print(f"  {key}: {path}")
        
        print(f"\n学習結果:")
        metrics = trainer.training_info['metrics']
        print(f"  テストMAE: ¥{metrics['test_mae']:,.0f}")
        print(f"  テストR²: {metrics['test_r2']:.3f}")
        print(f"  交差検証R²: {metrics['cv_r2_mean']:.3f}±{metrics['cv_r2_std']:.3f}")
        
        # 評価実行
        if args.evaluate:
            print(f"\n詳細評価を実行中...")
            try:
                from models.model_evaluator import ModelEvaluator
                evaluator = ModelEvaluator()
                evaluator.model = trainer.model
                evaluator.data = df_processed
                evaluator.encoders = trainer.encoders
                
                results = evaluator.evaluate_model_performance()
                print(f"価格帯別精度: {len(results['price_range_accuracy'])} 区分")
                print(f"特徴量重要度: {results['feature_importance']['features'][:3]}")
                
            except Exception as e:
                logger.warning(f"評価実行に失敗: {e}")
        
    except Exception as e:
        logger.error(f"学習中にエラーが発生: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()