#!/usr/bin/env python
"""
Docker環境互換モデル訓練スクリプト
NumPy 1.24.3, scikit-learn 1.3.2 環境で動作するモデルを作成
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time

# ロガー設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_docker_compatible_model():
    """Docker環境と互換性のあるモデルを訓練"""
    
    # バージョン確認
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    import sklearn
    logger.info(f"scikit-learn version: {sklearn.__version__}")
    
    # データ読み込み
    data_path = './model-creation/data/tokyo23_real_estate.csv'
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False
        
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # 必要な列のみ選択
    required_cols = ['Municipality', 'DistrictName', 'TradePrice', '土地面積', '建物面積', '築年数']
    df = df[required_cols].copy()
    
    # 欠損値処理
    df = df.dropna()
    logger.info(f"Data shape after cleaning: {df.shape}")
    
    # 特徴量とターゲット
    X = df[['Municipality', 'DistrictName', '土地面積', '建物面積', '築年数']].copy()
    y = df['TradePrice']
    
    # カテゴリカル変数のエンコーディング
    label_encoders = {}
    for col in ['Municipality', 'DistrictName']:
        le = LabelEncoder()
        X[f'{col}_encoded'] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # 数値特徴量のみ選択
    feature_cols = ['Municipality_encoded', 'DistrictName_encoded', '土地面積', '建物面積', '築年数']
    X_numeric = X[feature_cols]
    
    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 最適化されたモデル訓練（Docker環境用）
    logger.info("Training Docker-compatible Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=50,         # 軽量化
        max_depth=20,           
        min_samples_split=5,    
        min_samples_leaf=2,     
        max_features='sqrt',    
        n_jobs=1,              
        random_state=42
    )
    
    # 訓練時間の計測
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    logger.info(f"Model training completed in {train_time:.2f} seconds")
    
    # 評価
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    logger.info(f"Model Performance:")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:,.0f} 円")
    logger.info(f"  MAE: {mae:,.0f} 円")
    
    # 出力ディレクトリ作成
    output_dir = './docker_models'
    os.makedirs(output_dir, exist_ok=True)
    
    # モデル保存（Docker環境互換）
    model_path = os.path.join(output_dir, 'valuation_model.joblib')
    joblib.dump(model, model_path, compress=9)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    logger.info(f"Model saved to {model_path} ({model_size:.2f} MB)")
    
    # エンコーダー（日本語キーで保存）
    japanese_encoders = {
        '都道府県': None,  # Prefecture encoder not needed for Tokyo only
        '市区町村': label_encoders['Municipality'],
        '地区名': label_encoders['DistrictName']
    }
    encoders_path = os.path.join(output_dir, 'label_encoders.joblib')
    joblib.dump(japanese_encoders, encoders_path, compress=9)
    logger.info(f"Encoders saved to {encoders_path}")
    
    # スケーラー
    scaler_path = os.path.join(output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path, compress=9)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # 特徴量列
    feature_path = os.path.join(output_dir, 'feature_columns.joblib')
    joblib.dump(feature_cols, feature_path, compress=9)
    logger.info(f"Feature columns saved to {feature_path}")
    
    # バージョン情報保存
    version_info = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sklearn': sklearn.__version__,
        'model_type': 'RandomForestRegressor (Docker Compatible)',
        'n_estimators': 50,
        'max_depth': 20,
        'model_size_mb': model_size,
        'train_time_sec': train_time,
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'created_for': 'Docker Compose environment'
    }
    
    import json
    info_path = os.path.join(output_dir, 'docker_model_info.json')
    with open(info_path, 'w') as f:
        json.dump(version_info, f, indent=2)
    
    logger.info("Docker-compatible model creation completed!")
    logger.info(f"All files saved in: {output_dir}")
    
    # 検証テスト
    logger.info("Performing validation test...")
    try:
        # 再読み込みテスト
        test_model = joblib.load(model_path)
        test_encoders = joblib.load(encoders_path)
        test_scaler = joblib.load(scaler_path)
        
        # ダミー予測テスト
        dummy_input = np.array([[10, 5, 100, 80, 10]])
        dummy_scaled = test_scaler.transform(dummy_input)
        prediction = test_model.predict(dummy_scaled)[0]
        
        logger.info(f"✅ Validation successful - Test prediction: ¥{prediction:,.0f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = create_docker_compatible_model()
    sys.exit(0 if success else 1)