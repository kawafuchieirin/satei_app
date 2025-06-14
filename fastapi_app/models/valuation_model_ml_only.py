"""
統合的な不動産価格予測モデル（MLモデル専用版）
ルールベースロジックを完全に削除し、機械学習モデルのみを使用
"""

import logging
import joblib
import os
from pathlib import Path
from typing import Dict, Optional, List
import sys

# 親ディレクトリのパスを追加
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

# 機械学習モデルの利用可能性をチェック
ML_MODEL_AVAILABLE = False
try:
    from model_creation.train_ml_model import RealEstatePriceModel
    import numpy as np
    import pandas as pd
    ML_MODEL_AVAILABLE = True
    logger.info("ML model dependencies are available")
except ImportError as e:
    logger.error(f"ML model dependencies not available: {e}")


class ValuationModel:
    """統合的な不動産価格予測モデル（MLモデル専用）"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初期化
        
        Args:
            model_path: モデルファイルのパス
        """
        self.ml_model = None
        
        if not ML_MODEL_AVAILABLE:
            raise RuntimeError("機械学習の依存関係が利用できません。必要なパッケージをインストールしてください。")
        
        # MLモデルのロード
        try:
            # model_creationディレクトリのモデルを試す
            if model_path is None:
                model_creation_path = Path(__file__).parent.parent.parent / 'model_creation' / 'models'
                if model_creation_path.exists():
                    self.ml_model = RealEstatePriceModel(str(model_creation_path))
                    if hasattr(self.ml_model, 'load_model'):
                        self.ml_model.load_model()
                        logger.info("ML model loaded from model_creation directory")
                else:
                    # 現在のディレクトリのモデルを試す
                    current_model_path = Path(__file__).parent / 'ml_models'
                    if current_model_path.exists():
                        self.ml_model = RealEstatePriceModel(str(current_model_path))
                        if hasattr(self.ml_model, 'load_model'):
                            self.ml_model.load_model()
                            logger.info("ML model loaded from current directory")
            else:
                self.ml_model = RealEstatePriceModel(model_path)
                if hasattr(self.ml_model, 'load_model'):
                    self.ml_model.load_model()
                    logger.info(f"ML model loaded from {model_path}")
                    
            if self.ml_model is None:
                raise RuntimeError("MLモデルのロードに失敗しました")
                
        except Exception as e:
            logger.error(f"Failed to initialize ML model: {e}")
            raise RuntimeError(f"MLモデルの初期化に失敗しました: {str(e)}")
    
    def predict(self,
                prefecture: str,
                city: str,
                district: str,
                land_area: float,
                building_area: float,
                building_age: int) -> Dict:
        """
        不動産価格を予測（MLモデルのみ使用）
        
        Args:
            prefecture: 都道府県
            city: 市区町村
            district: 地区
            land_area: 土地面積（㎡）
            building_area: 建物面積（㎡）
            building_age: 築年数
            
        Returns:
            予測結果の辞書
        """
        # 入力検証
        if land_area <= 0:
            raise ValueError("土地面積は0より大きい値である必要があります")
        if building_area <= 0:
            raise ValueError("建物面積は0より大きい値である必要があります")
        if building_age < 0:
            raise ValueError("築年数は0以上である必要があります")
        
        # MLモデルでの予測
        if self.ml_model is None:
            raise RuntimeError("MLモデルが初期化されていません")
            
        try:
            return self.ml_model.predict(
                prefecture=prefecture,
                city=city,
                district=district,
                land_area=land_area,
                building_area=building_area,
                building_age=building_age
            )
        except Exception as e:
            logger.error(f"ML model prediction failed: {e}")
            # MLモデルが必須のため、エラーを再発生
            raise RuntimeError(f"機械学習モデルでの予測に失敗しました: {str(e)}")
    
    def _analyze_factors(self, 
                        land_area: float,
                        building_area: float,
                        building_age: int,
                        predicted_price: float) -> List[str]:
        """
        査定価格に影響する要因を分析
        """
        factors = []
        
        # 面積による影響
        if land_area > 150:
            factors.append("土地面積が広く、価格にプラス影響")
        elif land_area < 50:
            factors.append("土地面積が狭く、価格にマイナス影響")
        
        if building_area > 120:
            factors.append("建物面積が広く、価格にプラス影響")
        elif building_area < 60:
            factors.append("建物面積が狭く、価格にマイナス影響")
        
        # 築年数による影響
        if building_age <= 5:
            factors.append("築浅物件で、価格にプラス影響")
        elif building_age <= 15:
            factors.append("比較的新しい物件で、価格への影響は中程度")
        elif building_age <= 25:
            factors.append("築年数がやや古く、価格にマイナス影響")
        else:
            factors.append("築年数が古く、価格に大きなマイナス影響")
        
        # 価格レンジによる評価
        if predicted_price > 100000000:  # 1億円以上
            factors.append("高額物件として評価")
        elif predicted_price < 30000000:  # 3000万円未満
            factors.append("比較的手頃な価格帯の物件")
        
        return factors
    
    def get_model_info(self) -> Dict:
        """
        モデル情報を取得
        """
        if self.ml_model and hasattr(self.ml_model, 'model_info'):
            return self.ml_model.model_info
        else:
            return {
                'model_type': 'Machine Learning Required',
                'training_date': 'N/A',
                'accuracy': 'N/A',
                'version': '2.0.0'
            }