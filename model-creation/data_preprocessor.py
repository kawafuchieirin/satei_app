#!/usr/bin/env python3
"""
不動産データの前処理モジュール
MLITデータを機械学習モデル用に変換
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealEstateDataPreprocessor:
    """不動産データの前処理クラス"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.categorical_columns = ['都道府県', '市区町村', '地区名', '建物の構造', '用途']
        self.numerical_columns = ['土地面積', '建物面積', '築年数', '建ぺい率（％）', '容積率（％）']
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MLITデータから特徴量を準備
        """
        logger.info("Preparing features from raw data...")
        
        # 必要な列の確認と作成
        features = pd.DataFrame()
        
        # 基本情報
        features['都道府県'] = '東京都'  # 東京23区のデータなので固定
        features['市区町村'] = df['区'] if '区' in df.columns else df.get('市区町村名', '')
        features['地区名'] = df.get('地区名', '').fillna('不明')
        
        # 数値特徴量
        features['土地面積'] = df.get('土地面積', df.get('面積（㎡）', 0)).fillna(0)
        features['建物面積'] = df.get('建物面積', df.get('延床面積（㎡）', 0)).fillna(0)
        features['築年数'] = df.get('築年数', 0).fillna(0)
        
        # 建ぺい率・容積率
        features['建ぺい率（％）'] = df.get('建ぺい率（％）', 60).fillna(60)
        features['容積率（％）'] = df.get('容積率（％）', 200).fillna(200)
        
        # 建物構造
        features['建物の構造'] = df.get('建物の構造', 'RC').fillna('RC')
        features['用途'] = df.get('用途', '住宅').fillna('住宅')
        
        # 道路情報
        features['前面道路幅員'] = df.get('前面道路：幅員（ｍ）', 4.0).fillna(4.0)
        
        # 駅距離（データがない場合はダミー値）
        features['最寄駅距離'] = df.get('最寄駅：距離（分）', 10).fillna(10)
        
        # ターゲット変数
        features['取引価格'] = df['取引価格（総額）']
        
        # 価格の異常値を除外
        features = features[features['取引価格'] > 0]
        features = features[features['土地面積'] > 0]
        
        logger.info(f"Features prepared: {len(features)} records")
        
        return features
    
    def preprocess(self, 
                  df: pd.DataFrame, 
                  is_training: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        データの前処理とエンコーディング
        
        Args:
            df: 特徴量データ
            is_training: 訓練データかどうか
            
        Returns:
            X: 特徴量行列
            y: ターゲット変数
            feature_names: 特徴量名のリスト
        """
        # ターゲット変数を分離
        y = df['取引価格'].values if '取引価格' in df.columns else None
        X_df = df.drop(['取引価格'], axis=1, errors='ignore')
        
        # カテゴリカル変数のエンコーディング
        encoded_dfs = []
        
        for col in self.categorical_columns:
            if col in X_df.columns:
                if is_training:
                    # 訓練時は新しいエンコーダーを作成
                    le = LabelEncoder()
                    # Unknown categoryのために一つ余分なカテゴリを追加
                    unique_values = X_df[col].unique().tolist()
                    unique_values.append('__unknown__')
                    le.fit(unique_values)
                    self.label_encoders[col] = le
                    
                    # 実際のデータをエンコード
                    encoded = le.transform(X_df[col])
                else:
                    # 推論時は既存のエンコーダーを使用
                    le = self.label_encoders.get(col)
                    if le:
                        # 未知のカテゴリは__unknown__として扱う
                        encoded_values = []
                        for val in X_df[col]:
                            if val in le.classes_:
                                encoded_values.append(le.transform([val])[0])
                            else:
                                encoded_values.append(le.transform(['__unknown__'])[0])
                        encoded = np.array(encoded_values)
                    else:
                        encoded = np.zeros(len(X_df))
                
                encoded_df = pd.DataFrame({f'{col}_encoded': encoded})
                encoded_dfs.append(encoded_df)
        
        # 数値変数の選択
        numerical_dfs = []
        for col in self.numerical_columns:
            if col in X_df.columns:
                numerical_dfs.append(X_df[[col]])
        
        # 特徴量の結合
        all_features = encoded_dfs + numerical_dfs
        if all_features:
            X_processed = pd.concat(all_features, axis=1)
        else:
            X_processed = pd.DataFrame()
        
        # 特徴量名の記録
        self.feature_columns = X_processed.columns.tolist()
        
        # NumPy配列に変換
        X = X_processed.values
        
        # スケーリング
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        return X, y, self.feature_columns
    
    def create_model_features(self, 
                            prefecture: str,
                            city: str,
                            district: str,
                            land_area: float,
                            building_area: float,
                            building_age: int) -> np.ndarray:
        """
        モデル推論用の特徴量を作成
        """
        # 入力データをDataFrameに変換
        input_data = pd.DataFrame([{
            '都道府県': prefecture,
            '市区町村': city,
            '地区名': district,
            '土地面積': land_area,
            '建物面積': building_area,
            '築年数': building_age,
            '建ぺい率（％）': 60,  # デフォルト値
            '容積率（％）': 200,   # デフォルト値
            '建物の構造': 'RC',    # デフォルト値
            '用途': '住宅',        # デフォルト値
            '前面道路幅員': 4.0,   # デフォルト値
            '最寄駅距離': 10      # デフォルト値
        }])
        
        # 前処理（推論モード）
        X, _, _ = self.preprocess(input_data, is_training=False)
        
        return X
    
    def save_preprocessor(self, output_dir: str = 'models'):
        """
        前処理オブジェクトを保存
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Label Encodersの保存
        joblib.dump(self.label_encoders, output_path / 'label_encoders.joblib')
        
        # Scalerの保存
        joblib.dump(self.scaler, output_path / 'scaler.joblib')
        
        # 特徴量リストの保存
        joblib.dump(self.feature_columns, output_path / 'feature_columns.joblib')
        
        logger.info(f"Preprocessor saved to {output_path}")
    
    def load_preprocessor(self, model_dir: str = 'models'):
        """
        前処理オブジェクトを読み込み
        """
        model_path = Path(model_dir)
        
        # Label Encodersの読み込み
        encoder_path = model_path / 'label_encoders.joblib'
        if encoder_path.exists():
            self.label_encoders = joblib.load(encoder_path)
        
        # Scalerの読み込み
        scaler_path = model_path / 'scaler.joblib'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # 特徴量リストの読み込み
        features_path = model_path / 'feature_columns.joblib'
        if features_path.exists():
            self.feature_columns = joblib.load(features_path)
        
        logger.info(f"Preprocessor loaded from {model_path}")
    
    def get_feature_importance_names(self) -> List[str]:
        """
        特徴量の重要度表示用の名前を取得
        """
        readable_names = []
        
        for col in self.feature_columns:
            if col.endswith('_encoded'):
                # エンコードされたカテゴリカル変数
                original_name = col.replace('_encoded', '')
                readable_names.append(original_name)
            else:
                # 数値変数
                readable_names.append(col)
        
        return readable_names
    
    def split_data(self, 
                  X: np.ndarray, 
                  y: np.ndarray, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        データを訓練用とテスト用に分割
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    # 使用例
    preprocessor = RealEstateDataPreprocessor()
    
    # サンプルデータの作成
    sample_data = pd.DataFrame({
        '区': ['渋谷区', '新宿区', '港区'],
        '地区名': ['恵比寿', '歌舞伎町', '六本木'],
        '土地面積': [100, 120, 150],
        '建物面積': [80, 100, 120],
        '築年数': [10, 5, 15],
        '建ぺい率（％）': [60, 70, 80],
        '容積率（％）': [200, 300, 400],
        '建物の構造': ['RC', 'SRC', 'RC'],
        '用途': ['住宅', '共同住宅', '住宅'],
        '取引価格（総額）': [80000000, 120000000, 150000000]
    })
    
    # 特徴量の準備
    features_df = preprocessor.prepare_features(sample_data)
    
    # 前処理
    X, y, feature_names = preprocessor.preprocess(features_df, is_training=True)
    
    logger.info(f"Shape of X: {X.shape}")
    logger.info(f"Shape of y: {y.shape}")
    logger.info(f"Feature names: {feature_names}")