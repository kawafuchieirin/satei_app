#!/usr/bin/env python
"""
モデル予測のテストスクリプト
"""

import sys
sys.path.append('.')

from models.lightweight_model import LightweightValuationModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prediction():
    """ML予測のテスト"""
    try:
        # モデル初期化
        model = LightweightValuationModel()
        logger.info(f"Model available: {model.is_model_available()}")
        
        if not model.is_model_available():
            logger.error("Model not available")
            return False
        
        # テストデータ
        test_data = {
            "prefecture": "東京都",
            "city": "渋谷区", 
            "land_area": 100,
            "building_area": 80,
            "building_age": 10,
            "district": ""
        }
        
        logger.info(f"Testing with data: {test_data}")
        
        # 予測実行
        result = model.predict(**test_data)
        logger.info(f"Prediction result: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction()
    sys.exit(0 if success else 1)