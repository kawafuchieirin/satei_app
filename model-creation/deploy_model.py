#!/usr/bin/env python3
"""
訓練済みモデルをAPIサービスにデプロイするスクリプト
"""

import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_model_to_api():
    """訓練済みモデルをAPIディレクトリにコピー"""
    
    # パス設定
    model_dir = Path('models')
    api_dir = Path('../valuation-api')
    api_ml_dir = Path('../valuation-api-ml')
    
    # 必要なファイル
    files_to_copy = [
        'trained_model.joblib',  # 或いは 'quick_model.joblib'
        'label_encoders.joblib',
        'scaler.joblib',
        'feature_columns.joblib'
    ]
    
    success_count = 0
    
    # 各APIディレクトリにコピー
    for target_dir in [api_dir, api_ml_dir]:
        if not target_dir.exists():
            logger.warning(f"Target directory not found: {target_dir}")
            continue
            
        logger.info(f"Deploying to {target_dir}")
        
        for file_name in files_to_copy:
            src_file = model_dir / file_name
            
            if src_file.exists():
                # メインモデルは valuation_model.joblib として保存
                if file_name in ['trained_model.joblib', 'quick_model.joblib']:
                    dst_file = target_dir / 'valuation_model.joblib'
                else:
                    dst_file = target_dir / file_name
                
                shutil.copy2(src_file, dst_file)
                logger.info(f"  Copied: {file_name} -> {dst_file.name}")
                success_count += 1
            else:
                logger.warning(f"  Missing: {file_name}")
    
    if success_count > 0:
        logger.info(f"Model deployment completed! {success_count} files copied.")
        logger.info("モデルがAPIに統合されました。")
        logger.info("次の手順: APIを再デプロイしてください。")
    else:
        logger.error("No model files found. Please train a model first.")

def check_api_model_status():
    """APIディレクトリのモデルファイル状態をチェック"""
    
    api_dirs = [
        Path('../valuation-api'),
        Path('../valuation-api-ml')
    ]
    
    for api_dir in api_dirs:
        if not api_dir.exists():
            continue
            
        logger.info(f"\n=== {api_dir.name} Model Status ===")
        
        model_files = [
            'valuation_model.joblib',
            'label_encoders.joblib',
            'scaler.joblib',
            'feature_columns.joblib'
        ]
        
        for file_name in model_files:
            file_path = api_dir / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✅ {file_name} ({size_mb:.1f} MB)")
            else:
                logger.info(f"  ❌ {file_name} (Missing)")

if __name__ == "__main__":
    logger.info("=== Model Deployment Script ===")
    
    # 現在の状態をチェック
    check_api_model_status()
    
    # モデルをデプロイ
    deploy_model_to_api()
    
    # 結果を確認
    check_api_model_status()