#!/usr/bin/env python3
"""
MLモデル管理スクリプト
データ取得、モデル訓練、デプロイまでの一連のプロセスを管理
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tokyo23_data_fetcher import Tokyo23DataFetcher
from train_ml_model import RealEstatePriceModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelManager:
    """MLモデルの管理クラス"""
    
    def __init__(self):
        self.model_dir = Path('models')
        self.data_dir = Path('data')
        self.api_model_dir = Path('../valuation-api/models')
        
        # ディレクトリの作成
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
    
    def fetch_data(self, from_year: int = 2022, to_year: int = 2024):
        """
        MLIT APIから東京23区のデータを取得
        
        Returns:
            データファイルのパス
        """
        logger.info("Starting data collection from MLIT API...")
        
        # APIキーの確認
        api_key = os.getenv('MLIT_API_KEY')
        if not api_key:
            logger.warning("MLIT_API_KEY not found in environment variables")
            logger.info("Please set MLIT_API_KEY to fetch real data")
            logger.info("Continuing with sample data generation...")
        
        fetcher = Tokyo23DataFetcher(api_key=api_key)
        df = fetcher.fetch_all_tokyo23_data(
            from_year=from_year,
            to_year=to_year,
            save_csv=True
        )
        
        if df.empty:
            logger.error("No data fetched. Please check your API key.")
            return None
        
        # 最新のCSVファイルを取得
        csv_files = list(self.data_dir.glob('tokyo23_real_estate_*.csv'))
        if csv_files:
            latest_file = max(csv_files, key=os.path.getmtime)
            logger.info(f"Data saved to: {latest_file}")
            return str(latest_file)
        
        return None
    
    def train_model(self, data_path: str = None, fetch_new: bool = False):
        """
        モデルの訓練を実行
        """
        logger.info("Starting model training...")
        
        model = RealEstatePriceModel(model_dir=str(self.model_dir))
        
        results = model.train(
            data_path=data_path,
            fetch_new_data=fetch_new,
            test_size=0.2,
            cv_folds=5
        )
        
        logger.info("Model training completed!")
        logger.info(f"Test MAE: {results['test_metrics']['mae']:,.0f}")
        logger.info(f"Test R²: {results['test_metrics']['r2']:.3f}")
        
        return results
    
    def deploy_model(self, target: str = 'api'):
        """
        訓練済みモデルをAPIディレクトリにデプロイ
        
        Args:
            target: デプロイ先 ('api' or 'lambda')
        """
        logger.info(f"Deploying model to {target}...")
        
        # モデルファイルの確認
        model_files = [
            'real_estate_model.joblib',
            'label_encoders.joblib',
            'scaler.joblib',
            'feature_columns.joblib',
            'model_info.json'
        ]
        
        missing_files = []
        for file in model_files:
            if not (self.model_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing model files: {missing_files}")
            logger.error("Please train the model first.")
            return False
        
        # デプロイ先ディレクトリの作成
        if target == 'api':
            deploy_dir = self.api_model_dir
        else:
            deploy_dir = Path(f'../deployment/{target}_models')
        
        deploy_dir.mkdir(exist_ok=True, parents=True)
        
        # ファイルのコピー
        for file in model_files:
            src = self.model_dir / file
            dst = deploy_dir / file
            if src.exists():
                shutil.copy2(src, dst)
                logger.info(f"Copied {file} to {deploy_dir}")
        
        # デプロイ情報の記録
        deploy_info = {
            'deployed_at': datetime.now().isoformat(),
            'target': target,
            'model_files': model_files,
            'deploy_dir': str(deploy_dir)
        }
        
        with open(deploy_dir / 'deploy_info.json', 'w') as f:
            json.dump(deploy_info, f, indent=2)
        
        logger.info(f"Model deployed successfully to {deploy_dir}")
        return True
    
    def evaluate_deployed_model(self):
        """
        デプロイされたモデルの動作確認
        """
        logger.info("Evaluating deployed model...")
        
        try:
            # APIディレクトリのモデルを読み込み
            model = RealEstatePriceModel(model_dir=str(self.api_model_dir))
            model.load_model()
            
            # テスト予測
            test_cases = [
                {
                    'prefecture': '東京都',
                    'city': '渋谷区',
                    'district': '恵比寿',
                    'land_area': 100,
                    'building_area': 80,
                    'building_age': 10
                },
                {
                    'prefecture': '東京都',
                    'city': '港区',
                    'district': '六本木',
                    'land_area': 150,
                    'building_area': 120,
                    'building_age': 5
                }
            ]
            
            logger.info("Test predictions:")
            for i, test in enumerate(test_cases):
                result = model.predict(**test)
                logger.info(f"Test {i+1}:")
                logger.info(f"  Input: {test['city']} {test['district']}, "
                          f"土地{test['land_area']}㎡, 建物{test['building_area']}㎡, "
                          f"築{test['building_age']}年")
                logger.info(f"  Estimated price: ¥{result['estimated_price']:,.0f}")
                logger.info(f"  Confidence: {result['confidence']:.1f}%")
                logger.info(f"  Price range: ¥{result['price_range']['min']:,.0f} - "
                          f"¥{result['price_range']['max']:,.0f}")
            
            logger.info("Model evaluation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return False
    
    def full_pipeline(self):
        """
        データ取得からデプロイまでの完全なパイプラインを実行
        """
        logger.info("Starting full ML pipeline...")
        
        # 1. データ取得
        logger.info("Step 1: Fetching data...")
        data_path = self.fetch_data()
        
        if not data_path:
            logger.warning("Using sample data for training...")
            data_path = None
        
        # 2. モデル訓練
        logger.info("Step 2: Training model...")
        results = self.train_model(data_path=data_path)
        
        # 3. モデルデプロイ
        logger.info("Step 3: Deploying model...")
        success = self.deploy_model(target='api')
        
        if success:
            # 4. 動作確認
            logger.info("Step 4: Evaluating deployed model...")
            self.evaluate_deployed_model()
        
        logger.info("Full pipeline completed!")
        return success


def main():
    parser = argparse.ArgumentParser(
        description='ML Model Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (data fetch -> train -> deploy)
  python ml_model_manager.py full
  
  # Fetch data only
  python ml_model_manager.py fetch --from-year 2022 --to-year 2024
  
  # Train model with existing data
  python ml_model_manager.py train --data-path data/tokyo23_real_estate_2022_2024.csv
  
  # Deploy trained model
  python ml_model_manager.py deploy --target api
  
  # Evaluate deployed model
  python ml_model_manager.py evaluate
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Full pipeline
    subparsers.add_parser('full', help='Run full pipeline (fetch -> train -> deploy)')
    
    # Fetch data
    fetch_parser = subparsers.add_parser('fetch', help='Fetch data from MLIT API')
    fetch_parser.add_argument('--from-year', type=int, default=2022, help='Start year')
    fetch_parser.add_argument('--to-year', type=int, default=2024, help='End year')
    
    # Train model
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--data-path', type=str, help='Path to CSV data')
    train_parser.add_argument('--fetch-new', action='store_true', help='Fetch new data')
    
    # Deploy model
    deploy_parser = subparsers.add_parser('deploy', help='Deploy trained model')
    deploy_parser.add_argument('--target', choices=['api', 'lambda'], default='api',
                              help='Deployment target')
    
    # Evaluate model
    subparsers.add_parser('evaluate', help='Evaluate deployed model')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = MLModelManager()
    
    if args.command == 'full':
        manager.full_pipeline()
    elif args.command == 'fetch':
        manager.fetch_data(from_year=args.from_year, to_year=args.to_year)
    elif args.command == 'train':
        manager.train_model(data_path=args.data_path, fetch_new=args.fetch_new)
    elif args.command == 'deploy':
        manager.deploy_model(target=args.target)
    elif args.command == 'evaluate':
        manager.evaluate_deployed_model()


if __name__ == "__main__":
    main()