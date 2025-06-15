import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MLITDataFetcher:
    """
    国土交通省 不動産取引価格情報API からデータを取得するクラス
    """
    
    BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
    
    def __init__(self):
        self.session = requests.Session()
    
    def fetch_trade_data(self, 
                        prefecture: str, 
                        city: Optional[str] = None,
                        from_year: int = 2021,
                        to_year: int = 2024) -> pd.DataFrame:
        """
        指定された地域と期間の不動産取引データを取得
        
        Args:
            prefecture: 都道府県名
            city: 市区町村名（オプション）
            from_year: 開始年
            to_year: 終了年
            
        Returns:
            取引データのDataFrame
        """
        all_data = []
        
        for year in range(from_year, to_year + 1):
            for quarter in [1, 2, 3, 4]:
                try:
                    data = self._fetch_quarter_data(prefecture, city, year, quarter)
                    if data:
                        all_data.extend(data)
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {year}Q{quarter}: {e}")
        
        if not all_data:
            logger.warning("No data fetched from MLIT API")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        return self._clean_data(df)
    
    def _fetch_quarter_data(self, 
                           prefecture: str, 
                           city: Optional[str], 
                           year: int, 
                           quarter: int) -> List[Dict]:
        """
        指定された四半期のデータを取得
        """
        params = {
            'area': self._get_area_code(prefecture),
            'from': f"{year}{quarter:02d}",
            'to': f"{year}{quarter:02d}"
        }
        
        if city:
            params['city'] = city
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data:
                return data['data']
            else:
                logger.warning(f"No data field in response for {year}Q{quarter}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {year}Q{quarter}: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response for {year}Q{quarter}: {e}")
            return []
    
    def _get_area_code(self, prefecture: str) -> str:
        """
        都道府県名から地域コードを取得
        """
        area_codes = {
            '北海道': '01', '青森県': '02', '岩手県': '03', '宮城県': '04', '秋田県': '05',
            '山形県': '06', '福島県': '07', '茨城県': '08', '栃木県': '09', '群馬県': '10',
            '埼玉県': '11', '千葉県': '12', '東京都': '13', '神奈川県': '14', '新潟県': '15',
            '富山県': '16', '石川県': '17', '福井県': '18', '山梨県': '19', '長野県': '20',
            '岐阜県': '21', '静岡県': '22', '愛知県': '23', '三重県': '24', '滋賀県': '25',
            '京都府': '26', '大阪府': '27', '兵庫県': '28', '奈良県': '29', '和歌山県': '30',
            '鳥取県': '31', '島根県': '32', '岡山県': '33', '広島県': '34', '山口県': '35',
            '徳島県': '36', '香川県': '37', '愛媛県': '38', '高知県': '39', '福岡県': '40',
            '佐賀県': '41', '長崎県': '42', '熊本県': '43', '大分県': '44', '宮崎県': '45',
            '鹿児島県': '46', '沖縄県': '47'
        }
        
        return area_codes.get(prefecture, '13')  # デフォルトは東京都
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        取得したデータをクリーニング
        """
        if df.empty:
            return df
        
        # 必要な列のみ抽出（APIレスポンスの構造に依存）
        required_columns = ['取引価格（総額）', '面積（㎡）', '建築年', '所在地', '用途']
        
        # 実際の列名に合わせて調整
        available_columns = [col for col in required_columns if col in df.columns]
        
        if not available_columns:
            logger.warning("Required columns not found in data")
            return pd.DataFrame()
        
        df_clean = df[available_columns].copy()
        
        # 数値型への変換
        if '取引価格（総額）' in df_clean.columns:
            df_clean['取引価格（総額）'] = pd.to_numeric(df_clean['取引価格（総額）'], errors='coerce')
        
        if '面積（㎡）' in df_clean.columns:
            df_clean['面積（㎡）'] = pd.to_numeric(df_clean['面積（㎡）'], errors='coerce')
        
        # 欠損値の除去
        df_clean = df_clean.dropna()
        
        return df_clean
    
    def generate_sample_data(self) -> pd.DataFrame:
        """
        API が利用できない場合のサンプルデータを生成
        """
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        prefectures = ['東京都', '神奈川県', '大阪府', '愛知県', '埼玉県']
        cities = ['渋谷区', '新宿区', '港区', '千代田区', '中央区']
        districts = ['恵比寿', '青山', '赤坂', '銀座', '丸の内']
        
        data = []
        for _ in range(n_samples):
            land_area = np.random.normal(100, 30)
            building_area = np.random.normal(80, 25)
            building_age = np.random.randint(0, 50)
            
            # 価格計算（サンプル）
            base_price = 500000  # 平米単価
            location_factor = np.random.uniform(0.8, 2.0)
            age_factor = max(0.3, 1.0 - building_age * 0.02)
            
            price = (land_area * base_price * location_factor + 
                    building_area * base_price * 0.6 * age_factor)
            
            data.append({
                '都道府県': np.random.choice(prefectures),
                '市区町村': np.random.choice(cities),
                '地区': np.random.choice(districts),
                '土地面積': max(20, land_area),
                '建物面積': max(20, building_area),
                '築年数': building_age,
                '取引価格': int(price)
            })
        
        return pd.DataFrame(data)