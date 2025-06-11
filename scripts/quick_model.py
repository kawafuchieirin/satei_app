#!/usr/bin/env python3
"""
クイックモデル作成スクリプト

最小限の設定で高品質なモデルを素早く作成します。

使用方法:
python quick_model.py [--fast|--balanced|--best]
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QuickModelCreator:
    """
    クイックモデル作成クラス
    """
    
    def __init__(self, preset='balanced', output_dir='../api'):
        self.preset = preset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = 42
        
        # プリセット設定
        self.presets = {
            'fast': {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 5,
                'description': '高速学習（精度: 普通、速度: 高速）'
            },
            'balanced': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 2,
                'description': 'バランス型（精度: 良好、速度: 普通）'
            },
            'best': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 2,
                'description': '高精度（精度: 最高、速度: 低速）'
            }
        }
    
    def generate_sample_data(self, n_samples=1500):
        """
        高品質サンプルデータの生成
        """
        np.random.seed(self.random_state)
        
        # より現実的な地域設定
        prefectures = ['東京都', '神奈川県', '大阪府', '愛知県', '埼玉県', '千葉県']
        cities = {
            '東京都': ['渋谷区', '新宿区', '港区', '千代田区', '中央区', '世田谷区', '目黒区'],
            '神奈川県': ['横浜市', '川崎市', '相模原市', '藤沢市'],
            '大阪府': ['大阪市', '堺市', '東大阪市', '豊中市'],
            '愛知県': ['名古屋市', '豊田市', '岡崎市'],
            '埼玉県': ['さいたま市', '川口市', '所沢市'],
            '千葉県': ['千葉市', '船橋市', '柏市']
        }
        
        districts = ['中央', '東', '西', '南', '北', '駅前', '新町', '本町']
        
        # 地域別価格係数（より現実的）
        price_multipliers = {
            '東京都': {'渋谷区': 2.8, '新宿区': 2.5, '港区': 3.2, '千代田区': 3.5, 
                     '中央区': 3.0, '世田谷区': 2.2, '目黒区': 2.4},
            '神奈川県': {'横浜市': 1.8, '川崎市': 1.6, '相模原市': 1.3, '藤沢市': 1.5},
            '大阪府': {'大阪市': 1.7, '堺市': 1.3, '東大阪市': 1.2, '豊中市': 1.4},
            '愛知県': {'名古屋市': 1.5, '豊田市': 1.2, '岡崎市': 1.1},
            '埼玉県': {'さいたま市': 1.4, '川口市': 1.3, '所沢市': 1.2},
            '千葉県': {'千葉市': 1.3, '船橋市': 1.4, '柏市': 1.2}
        }
        
        data = []
        
        for _ in range(n_samples):
            prefecture = np.random.choice(prefectures)
            city = np.random.choice(list(cities[prefecture]))
            district = np.random.choice(districts)
            
            # 面積の生成（対数正規分布）
            land_area = np.random.lognormal(4.6, 0.5)  # 平均約100㎡
            building_area = land_area * np.random.uniform(0.5, 1.1)
            
            # 築年数（ワイブル分布で現実的な分布）
            building_age = np.random.weibull(2) * 25
            building_age = min(building_age, 50)
            
            # 価格計算（より精密な計算式）
            base_price_per_sqm = 400000  # 基準価格
            
            # 地域係数
            location_factor = price_multipliers[prefecture][city]
            location_factor *= np.random.uniform(0.9, 1.1)  # 地域内のばらつき
            
            # 面積効果
            area_factor = np.sqrt(land_area + building_area * 0.8)
            
            # 築年劣化
            age_factor = max(0.4, np.exp(-building_age * 0.03))
            
            # 市場ランダム要因
            market_factor = np.random.lognormal(0, 0.2)
            
            price = (base_price_per_sqm * area_factor * location_factor * 
                    age_factor * market_factor)
            
            # 特殊物件（5%の確率）
            if np.random.random() < 0.05:
                price *= np.random.uniform(1.5, 2.5)
            
            data.append({
                '都道府県': prefecture,
                '市区町村': city,
                '地区': district,
                '土地面積': max(25, land_area),
                '建物面積': max(20, building_area),
                '築年数': int(building_age),
                '取引価格': int(price)
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """
        データ前処理
        """
        logger.info("データ前処理中...")
        
        df_clean = df.copy()
        original_count = len(df_clean)
        
        # 異常値除去
        df_clean = df_clean[
            (df_clean['取引価格'] > 5_000_000) &
            (df_clean['取引価格'] < 2_000_000_000) &
            (df_clean['土地面積'] > 15) &
            (df_clean['土地面積'] < 1000) &
            (df_clean['建物面積'] > 15) &
            (df_clean['建物面積'] < 800) &
            (df_clean['築年数'] >= 0) &
            (df_clean['築年数'] <= 60)
        ]
        
        # 欠損値除去
        df_clean = df_clean.dropna()
        
        removed_count = original_count - len(df_clean)
        logger.info(f"前処理完了: {removed_count} 件削除 ({removed_count/original_count*100:.1f}%)")
        
        return df_clean
    
    def prepare_features(self, df):
        """
        特徴量準備
        """
        feature_columns = ['都道府県', '市区町村', '地区', '土地面積', '建物面積', '築年数']
        X = df[feature_columns].copy()
        y = df['取引価格']
        
        # カテゴリカル変数のエンコーディング
        categorical_columns = ['都道府県', '市区町村', '地区']
        self.encoders = {}
        
        for col in categorical_columns:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            self.encoders[col] = encoder
        
        return X, y
    
    def create_and_train_model(self, X, y):
        """
        モデル作成・学習
        """
        config = self.presets[self.preset]
        logger.info(f"モデル学習中: {config['description']}")
        
        # モデル作成
        model = RandomForestRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 訓練・テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # 学習
        model.fit(X_train, y_train)
        
        # 評価
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100
        }
        
        # 交差検証
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        return model, metrics, (X_train, X_test, y_train, y_test, y_test_pred)
    
    def save_model(self, model, metrics, data_info):
        """
        モデル保存
        """
        logger.info("モデル保存中...")
        
        # モデル保存
        model_path = self.output_dir / 'valuation_model.joblib'
        joblib.dump(model, model_path)
        
        # エンコーダー保存
        encoders_path = self.output_dir / 'label_encoders.joblib'
        joblib.dump(self.encoders, encoders_path)
        
        # 学習情報保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        training_info = {
            'timestamp': timestamp,
            'preset': self.preset,
            'preset_description': self.presets[self.preset]['description'],
            'model_params': self.presets[self.preset],
            'metrics': {k: float(v) for k, v in metrics.items()},
            'data_info': data_info
        }
        
        info_path = self.output_dir / f'quick_model_info_{timestamp}.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存完了:")
        logger.info(f"  モデル: {model_path}")
        logger.info(f"  エンコーダー: {encoders_path}")
        logger.info(f"  情報: {info_path}")
        
        return model_path, encoders_path, info_path
    
    def run(self):
        """
        クイックモデル作成の実行
        """
        print("=" * 60)
        print("⚡ クイックモデル作成")
        print("=" * 60)
        print(f"プリセット: {self.preset}")
        print(f"説明: {self.presets[self.preset]['description']}")
        print()
        
        try:
            # データ生成
            logger.info("サンプルデータ生成中...")
            df = self.generate_sample_data()
            logger.info(f"生成完了: {len(df)} 件")
            
            # データ前処理
            df_clean = self.preprocess_data(df)
            
            # 特徴量準備
            X, y = self.prepare_features(df_clean)
            
            # モデル学習
            model, metrics, split_data = self.create_and_train_model(X, y)
            
            # 結果表示
            print("📊 学習結果:")
            print("-" * 40)
            print(f"Test R² Score:  {metrics['test_r2']:.3f}")
            print(f"Test MAE:       ¥{metrics['test_mae']:,.0f}")
            print(f"Test MAPE:      {metrics['test_mape']:.1f}%")
            print(f"CV R² Score:    {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
            print()
            
            # データ情報
            data_info = {
                'total_samples': len(df_clean),
                'train_samples': len(split_data[0]),
                'test_samples': len(split_data[1])
            }
            
            # モデル保存
            paths = self.save_model(model, metrics, data_info)
            
            # 成功メッセージ
            if metrics['test_r2'] >= 0.7:
                status = "🌟 優秀"
                message = "高精度なモデルが作成されました！"
            elif metrics['test_r2'] >= 0.5:
                status = "👍 良好"
                message = "実用的なモデルが作成されました。"
            else:
                status = "⚠️ 注意"
                message = "精度にやや改善の余地があります。"
            
            print("=" * 60)
            print(f"🎉 モデル作成完了！")
            print("=" * 60)
            print(f"評価: {status}")
            print(f"メッセージ: {message}")
            print()
            print("💡 次のステップ:")
            print("1. docker-compose restart api  # APIサーバー再起動")
            print("2. python test_model_accuracy.py  # モデルテスト")
            print("3. ./manage_model.sh evaluate  # 詳細評価")
            
            return True
            
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='クイックモデル作成スクリプト')
    parser.add_argument('--preset', choices=['fast', 'balanced', 'best'], 
                       default='balanced', help='学習プリセット')
    parser.add_argument('--output-dir', default='../api', 
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # クイックモデル作成実行
    creator = QuickModelCreator(
        preset=args.preset,
        output_dir=args.output_dir
    )
    
    success = creator.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()