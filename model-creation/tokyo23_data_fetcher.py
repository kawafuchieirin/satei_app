#!/usr/bin/env python3
"""
東京23区専用の不動産取引データ取得モジュール
国土交通省 不動産取引価格情報APIから2022-2024年のデータを収集
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tokyo23DataFetcher:
    """東京23区の不動産取引データを取得するクラス"""
    
    # 国土交通省 不動産取引価格情報API エンドポイント
    BASE_URL = "https://www.land.mlit.go.jp/webland/api/TradeListSearch"
    
    # 東京23区の市区町村コード
    TOKYO_23_WARDS = {
        '千代田区': '13101',
        '中央区': '13102',
        '港区': '13103',
        '新宿区': '13104',
        '文京区': '13105',
        '台東区': '13106',
        '墨田区': '13107',
        '江東区': '13108',
        '品川区': '13109',
        '目黒区': '13110',
        '大田区': '13111',
        '世田谷区': '13112',
        '渋谷区': '13113',
        '中野区': '13114',
        '杉並区': '13115',
        '豊島区': '13116',
        '北区': '13117',
        '荒川区': '13118',
        '板橋区': '13119',
        '練馬区': '13120',
        '足立区': '13121',
        '葛飾区': '13122',
        '江戸川区': '13123'
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初期化
        
        Args:
            api_key: MLIT APIキー（環境変数MLIT_API_KEYからも取得可能）
        """
        self.api_key = api_key or os.getenv('MLIT_API_KEY')
        self.session = requests.Session()
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
    def fetch_all_tokyo23_data(self, 
                              from_year: int = 2022, 
                              to_year: int = 2024,
                              save_csv: bool = True) -> pd.DataFrame:
        """
        東京23区全ての取引データを取得
        
        Args:
            from_year: 開始年
            to_year: 終了年
            save_csv: CSVファイルとして保存するか
            
        Returns:
            全取引データのDataFrame
        """
        all_data = []
        
        for ward_name, city_code in self.TOKYO_23_WARDS.items():
            logger.info(f"Fetching data for {ward_name} (code: {city_code})")
            
            try:
                ward_data = self._fetch_ward_data(
                    ward_name=ward_name,
                    city_code=city_code,
                    from_year=from_year,
                    to_year=to_year
                )
                
                if not ward_data.empty:
                    all_data.append(ward_data)
                    logger.info(f"  → {len(ward_data)} records fetched")
                else:
                    logger.warning(f"  → No data found for {ward_name}")
                    
                # APIレート制限対策
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {ward_name}: {e}")
                continue
        
        if not all_data:
            logger.error("No data fetched from any ward")
            return pd.DataFrame()
        
        # 全データを結合
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total records fetched: {len(df)}")
        
        # データクリーニング
        df = self._clean_and_prepare_data(df)
        
        # CSV保存
        if save_csv:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.data_dir / f'tokyo23_real_estate_{from_year}_{to_year}_{timestamp}.csv'
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"Data saved to: {filename}")
        
        return df
    
    def _fetch_ward_data(self, 
                        ward_name: str, 
                        city_code: str,
                        from_year: int,
                        to_year: int) -> pd.DataFrame:
        """
        特定の区のデータを取得
        """
        all_records = []
        
        for year in range(from_year, to_year + 1):
            for quarter in range(1, 5):
                try:
                    params = {
                        'from': f"{year}{quarter:01d}",
                        'to': f"{year}{quarter:01d}",
                        'city': city_code
                    }
                    
                    # APIキーがある場合は追加
                    if self.api_key:
                        params['apikey'] = self.api_key
                    
                    response = self.session.get(
                        self.BASE_URL,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'data' in data and data['data']:
                            records = data['data']
                            # 区名を追加
                            for record in records:
                                record['区'] = ward_name
                            all_records.extend(records)
                    else:
                        logger.warning(f"API returned status {response.status_code} for {year}Q{quarter}")
                        
                except Exception as e:
                    logger.warning(f"Failed to fetch {ward_name} {year}Q{quarter}: {e}")
                    continue
                
                # レート制限対策
                time.sleep(0.5)
        
        if all_records:
            return pd.DataFrame(all_records)
        else:
            return pd.DataFrame()
    
    def _clean_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データのクリーニングと前処理
        """
        if df.empty:
            return df
        
        logger.info("Cleaning and preparing data...")
        
        # 必要な列を抽出（APIレスポンスに依存）
        # 注: 実際のAPIレスポンスの列名に合わせて調整が必要
        column_mapping = {
            'Type': '取引の種類',
            'Region': '地域',
            'MunicipalityCode': '市区町村コード',
            'Prefecture': '都道府県名',
            'Municipality': '市区町村名',
            'DistrictName': '地区名',
            'TradePrice': '取引価格（総額）',
            'Area': '面積（㎡）',
            'LandShape': '土地の形状',
            'Frontage': '間口',
            'TotalFloorArea': '延床面積（㎡）',
            'BuildingYear': '建築年',
            'Structure': '建物の構造',
            'Use': '用途',
            'Purpose': '今後の利用目的',
            'Direction': '前面道路：方位',
            'Classification': '前面道路：種類',
            'Breadth': '前面道路：幅員（ｍ）',
            'CityPlanning': '都市計画',
            'CoverageRatio': '建ぺい率（％）',
            'FloorAreaRatio': '容積率（％）',
            'Period': '取引時点'
        }
        
        # 利用可能な列のみを使用
        available_columns = {}
        for eng, jpn in column_mapping.items():
            if eng in df.columns:
                available_columns[eng] = jpn
        
        if available_columns:
            df = df.rename(columns=available_columns)
        
        # 住宅・マンションのみを抽出（用途でフィルタリング）
        if '用途' in df.columns:
            df = df[df['用途'].isin(['住宅', '共同住宅', '店舗兼住宅', '事務所兼住宅'])]
        
        # 数値データの変換
        numeric_columns = ['取引価格（総額）', '面積（㎡）', '延床面積（㎡）', 
                          '建ぺい率（％）', '容積率（％）']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 築年数の計算
        if '建築年' in df.columns:
            current_year = datetime.now().year
            df['建築年_西暦'] = df['建築年'].apply(self._convert_to_western_year)
            df['築年数'] = current_year - df['建築年_西暦']
            df['築年数'] = df['築年数'].clip(lower=0)  # 負の値を0に
        
        # 土地面積と建物面積の分離（必要に応じて）
        if '面積（㎡）' in df.columns and '延床面積（㎡）' in df.columns:
            df['土地面積'] = df['面積（㎡）']
            df['建物面積'] = df['延床面積（㎡）']
        
        # 欠損値の処理
        df = df.dropna(subset=['取引価格（総額）'])
        
        # 異常値の除外（価格が極端に高い/低いものを除外）
        if '取引価格（総額）' in df.columns:
            price_lower = df['取引価格（総額）'].quantile(0.01)
            price_upper = df['取引価格（総額）'].quantile(0.99)
            df = df[(df['取引価格（総額）'] >= price_lower) & 
                   (df['取引価格（総額）'] <= price_upper)]
        
        logger.info(f"Data after cleaning: {len(df)} records")
        
        return df
    
    def _convert_to_western_year(self, japanese_year: str) -> int:
        """
        和暦を西暦に変換
        """
        if pd.isna(japanese_year):
            return datetime.now().year
        
        try:
            # 令和の処理
            if '令和' in str(japanese_year):
                year_num = int(''.join(filter(str.isdigit, str(japanese_year))))
                return 2019 + year_num - 1
            # 平成の処理
            elif '平成' in str(japanese_year):
                year_num = int(''.join(filter(str.isdigit, str(japanese_year))))
                return 1989 + year_num - 1
            # 昭和の処理
            elif '昭和' in str(japanese_year):
                year_num = int(''.join(filter(str.isdigit, str(japanese_year))))
                return 1926 + year_num - 1
            # 既に西暦の場合
            else:
                return int(''.join(filter(str.isdigit, str(japanese_year))))
        except:
            return datetime.now().year
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        データの要約統計を取得
        """
        if df.empty:
            return {}
        
        stats = {
            'total_records': len(df),
            'wards_included': df['区'].nunique() if '区' in df.columns else 0,
            'price_stats': {
                'mean': df['取引価格（総額）'].mean() if '取引価格（総額）' in df.columns else 0,
                'median': df['取引価格（総額）'].median() if '取引価格（総額）' in df.columns else 0,
                'std': df['取引価格（総額）'].std() if '取引価格（総額）' in df.columns else 0,
                'min': df['取引価格（総額）'].min() if '取引価格（総額）' in df.columns else 0,
                'max': df['取引価格（総額）'].max() if '取引価格（総額）' in df.columns else 0
            }
        }
        
        if '区' in df.columns:
            stats['records_by_ward'] = df['区'].value_counts().to_dict()
        
        return stats


if __name__ == "__main__":
    # 使用例
    fetcher = Tokyo23DataFetcher()
    
    # APIキーが必要な場合は環境変数MLIT_API_KEYに設定するか、
    # fetcher = Tokyo23DataFetcher(api_key="your_api_key") として初期化
    
    # データ取得
    logger.info("Starting data collection for Tokyo 23 wards...")
    df = fetcher.fetch_all_tokyo23_data(from_year=2022, to_year=2024, save_csv=True)
    
    if not df.empty:
        # 統計情報の表示
        stats = fetcher.get_summary_statistics(df)
        logger.info(f"Summary statistics: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    else:
        logger.error("No data collected. Please check your API key and connection.")