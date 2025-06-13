#!/usr/bin/env python3
"""
東京23区不動産サンプルデータ生成スクリプト
MLモデル学習用の現実的な不動産取引データを生成
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

# 東京23区の基準価格と特性
WARD_DATA = {
    '千代田区': {'base_price': 250, 'premium': 1.5, 'volume': 50},
    '中央区': {'base_price': 200, 'premium': 1.3, 'volume': 100},
    '港区': {'base_price': 220, 'premium': 1.4, 'volume': 150},
    '新宿区': {'base_price': 150, 'premium': 1.2, 'volume': 200},
    '文京区': {'base_price': 140, 'premium': 1.1, 'volume': 120},
    '台東区': {'base_price': 120, 'premium': 1.0, 'volume': 90},
    '墨田区': {'base_price': 100, 'premium': 0.9, 'volume': 80},
    '江東区': {'base_price': 110, 'premium': 1.0, 'volume': 130},
    '品川区': {'base_price': 160, 'premium': 1.2, 'volume': 180},
    '目黒区': {'base_price': 170, 'premium': 1.3, 'volume': 140},
    '大田区': {'base_price': 130, 'premium': 1.0, 'volume': 220},
    '世田谷区': {'base_price': 140, 'premium': 1.1, 'volume': 300},
    '渋谷区': {'base_price': 180, 'premium': 1.3, 'volume': 160},
    '中野区': {'base_price': 120, 'premium': 1.0, 'volume': 150},
    '杉並区': {'base_price': 130, 'premium': 1.0, 'volume': 200},
    '豊島区': {'base_price': 140, 'premium': 1.1, 'volume': 140},
    '北区': {'base_price': 100, 'premium': 0.9, 'volume': 120},
    '荒川区': {'base_price': 90, 'premium': 0.8, 'volume': 80},
    '板橋区': {'base_price': 100, 'premium': 0.9, 'volume': 180},
    '練馬区': {'base_price': 110, 'premium': 0.9, 'volume': 200},
    '足立区': {'base_price': 80, 'premium': 0.8, 'volume': 150},
    '葛飾区': {'base_price': 85, 'premium': 0.8, 'volume': 120},
    '江戸川区': {'base_price': 90, 'premium': 0.8, 'volume': 140}
}

def generate_sample_data(num_samples: int = 5000) -> pd.DataFrame:
    """
    現実的な東京23区不動産取引サンプルデータを生成
    
    Args:
        num_samples: 生成するサンプル数
        
    Returns:
        pd.DataFrame: 不動産取引データ
    """
    data = []
    
    for _ in range(num_samples):
        # ランダムに区を選択（人口・取引量を考慮した重み付け）
        ward = random.choices(
            list(WARD_DATA.keys()),
            weights=[info['volume'] for info in WARD_DATA.values()],
            k=1
        )[0]
        
        ward_info = WARD_DATA[ward]
        base_price = ward_info['base_price']
        premium = ward_info['premium']
        
        # 基本データ生成
        land_area = np.random.lognormal(mean=4.0, sigma=0.8)  # 50-200㎡程度
        land_area = max(30, min(500, land_area))  # 範囲制限
        
        building_area = land_area * np.random.uniform(0.6, 1.2)  # 建ぺい率60-120%
        building_area = max(20, min(400, building_area))
        
        building_age = np.random.exponential(scale=15)  # 築年数（指数分布）
        building_age = max(0, min(50, building_age))
        
        # 価格計算（現実的なロジック）
        # 土地価格
        land_price = land_area * base_price * premium
        
        # 建物価格（減価あり）
        building_depreciation = max(0.2, 1 - (building_age * 0.02))
        building_price = building_area * base_price * 0.6 * building_depreciation
        
        # 総価格（万円→円）
        total_price = (land_price + building_price) * 10000
        
        # 市場変動要因
        market_factor = np.random.normal(1.0, 0.15)  # ±15%の市場変動
        total_price *= market_factor
        
        # 立地補正（駅距離等）
        location_factor = np.random.uniform(0.8, 1.3)
        total_price *= location_factor
        
        # 取引時期（2022-2024年）
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        
        # データレコード作成
        record = {
            'prefecture': '東京都',
            'city': ward,
            'district': f'{ward[:2]}地区',  # 簡略化した地区名
            'land_area': round(land_area, 1),
            'building_area': round(building_area, 1),
            'building_age': round(building_age, 1),
            'trade_price': round(total_price),
            'trade_date': random_date.strftime('%Y-%m-%d'),
            'quarter': f"{random_date.year}Q{(random_date.month-1)//3 + 1}"
        }
        
        data.append(record)
    
    df = pd.DataFrame(data)
    
    # データ品質向上
    # 外れ値除去（価格が極端なもの）
    price_q1 = df['trade_price'].quantile(0.01)
    price_q99 = df['trade_price'].quantile(0.99)
    df = df[(df['trade_price'] >= price_q1) & (df['trade_price'] <= price_q99)]
    
    return df

def main():
    """メイン実行"""
    print("東京23区不動産サンプルデータを生成中...")
    
    # データ生成
    df = generate_sample_data(num_samples=5000)
    
    # 保存
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / 'tokyo23_sample_data.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ サンプルデータ生成完了: {output_path}")
    print(f"📊 データ件数: {len(df):,}件")
    print(f"💰 価格範囲: {df['trade_price'].min():,.0f}円 - {df['trade_price'].max():,.0f}円")
    print(f"🏠 平均価格: {df['trade_price'].mean():,.0f}円")
    
    # 区別統計
    print("\n📈 区別データ統計:")
    ward_stats = df.groupby('city').agg({
        'trade_price': ['count', 'mean'],
        'land_area': 'mean',
        'building_age': 'mean'
    }).round(1)
    
    ward_stats.columns = ['件数', '平均価格', '平均土地面積', '平均築年数']
    print(ward_stats)

if __name__ == "__main__":
    main()