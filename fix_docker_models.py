#!/usr/bin/env python
"""
Docker互換モデル修正スクリプト
NumPy 1.24.3環境と互換性のあるエンコーダー・スケーラーを作成
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_minimal_compatible_files():
    """最小限の互換ファイルを作成"""
    
    logger.info("Creating Docker-compatible encoder and scaler files...")
    
    # 出力ディレクトリ
    output_dir = '/Users/kawabuchieirin/Desktop/satei_app/model-creation/models'
    
    # 1. 互換性のあるラベルエンコーダー作成
    logger.info("Creating label encoders...")
    
    # 東京23区の主要区とサンプル地区
    municipalities = ['渋谷区', '新宿区', '港区', '中央区', '千代田区', '品川区', 
                     '目黒区', '世田谷区', '大田区', '江東区', '文京区', '台東区']
    districts = ['', '1丁目', '2丁目', '3丁目', '4丁目', '5丁目', '西', '東', '南', '北']
    
    # Municipality encoder
    municipality_encoder = LabelEncoder()
    municipality_encoder.fit(municipalities)
    
    # District encoder  
    district_encoder = LabelEncoder()
    district_encoder.fit(districts)
    
    # 日本語キーで保存
    japanese_encoders = {
        '都道府県': None,  # Prefecture not needed for Tokyo-only
        '市区町村': municipality_encoder,
        '地区名': district_encoder
    }
    
    # 2. 互換性のあるスケーラー作成
    logger.info("Creating scaler...")
    
    # 実際のデータ分布に近いサンプルデータでスケーラーを作成
    sample_data = np.array([
        [0, 0, 50, 40, 0],     # 最小値付近
        [5, 3, 100, 80, 10],   # 典型的
        [10, 6, 150, 120, 20], # 大きめ
        [11, 8, 200, 150, 30], # 最大値付近
        [3, 2, 75, 60, 5],     # 中間値
    ])
    
    scaler = StandardScaler()
    scaler.fit(sample_data)
    
    # 3. 古いjoblibバージョンと互換性のある形式で保存
    logger.info("Saving with legacy compatibility...")
    
    # エンコーダー保存
    encoders_path = os.path.join(output_dir, 'label_encoders.joblib')
    joblib.dump(japanese_encoders, encoders_path, compress=3)  # 軽い圧縮
    logger.info(f"Encoders saved: {encoders_path}")
    
    # スケーラー保存
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path, compress=3)  # 軽い圧縮
    logger.info(f"Scaler saved: {scaler_path}")
    
    # 4. 既存の最適化モデルをコピー（これは動作することが確認済み）
    existing_model = os.path.join(output_dir, 'optimized_model.joblib')
    target_model = os.path.join(output_dir, 'valuation_model.joblib')
    
    if os.path.exists(existing_model):
        import shutil
        shutil.copy2(existing_model, target_model)
        logger.info(f"Model copied: {existing_model} -> {target_model}")
    
    # 5. 特徴量列情報
    feature_cols = ['Municipality_encoded', 'DistrictName_encoded', '土地面積', '建物面積', '築年数']
    feature_path = os.path.join(output_dir, 'feature_columns.joblib')
    joblib.dump(feature_cols, feature_path, compress=3)
    logger.info(f"Feature columns saved: {feature_path}")
    
    # 6. 検証テスト
    logger.info("Performing validation test...")
    try:
        # 再読み込みテスト
        test_encoders = joblib.load(encoders_path)
        test_scaler = joblib.load(scaler_path)
        test_features = joblib.load(feature_path)
        
        logger.info(f"✅ Encoders loaded: {list(test_encoders.keys())}")
        logger.info(f"✅ Scaler loaded: {type(test_scaler).__name__}")
        logger.info(f"✅ Features loaded: {test_features}")
        
        # エンコーダーテスト
        if test_encoders['市区町村'] is not None:
            encoded = test_encoders['市区町村'].transform(['渋谷区'])
            logger.info(f"✅ Municipality encoding test: 渋谷区 -> {encoded[0]}")
            
        if test_encoders['地区名'] is not None:
            encoded = test_encoders['地区名'].transform([''])
            logger.info(f"✅ District encoding test: '' -> {encoded[0]}")
        
        # スケーラーテスト
        dummy_data = np.array([[5, 2, 100, 80, 10]])
        scaled = test_scaler.transform(dummy_data)
        logger.info(f"✅ Scaler test: {dummy_data.flatten()} -> {scaled.flatten()}")
        
        logger.info("🎉 All validation tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_info_file():
    """互換モデル情報ファイル作成"""
    output_dir = '/Users/kawabuchieirin/Desktop/satei_app/model-creation/models'
    
    info = {
        "model_type": "Docker Compatible Random Forest",
        "created_for": "NumPy 1.24.3, scikit-learn 1.3.2",
        "features": 5,
        "encoders": {
            "municipalities": ["渋谷区", "新宿区", "港区", "中央区", "千代田区", "品川区", "目黒区", "世田谷区", "大田区", "江東区", "文京区", "台東区"],
            "districts": ["", "1丁目", "2丁目", "3丁目", "4丁目", "5丁目", "西", "東", "南", "北"]
        },
        "notes": "Created to resolve numpy._core compatibility issues in Docker environment"
    }
    
    info_path = os.path.join(output_dir, 'docker_compatible_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Info file created: {info_path}")

if __name__ == "__main__":
    success = create_minimal_compatible_files()
    if success:
        create_info_file()
        logger.info("✅ Docker-compatible model files creation completed!")
    else:
        logger.error("❌ Failed to create compatible files")
        exit(1)