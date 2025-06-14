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
import socket
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tokyo23DataFetcher:
    """東京23区の不動産取引データを取得するクラス"""
    
    # 国土交通省 不動産取引価格情報API エンドポイント
    BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
    
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
    
    def __init__(self):
        """
        初期化
        
        環境変数MLIT_API_KEYからAPIキーを取得します
        """
        self.api_key = os.getenv('MLIT_API_KEY')
        
        # リトライ設定付きのセッションを作成
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # タイムアウトとヘッダー設定
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'ja,en;q=0.9'
        })
        
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        # DNS問題のチェック
        self._check_connection()
        
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
        failed_wards = []
        
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
                    failed_wards.append(ward_name)
                    
                # APIレート制限対策
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {ward_name}: {e}")
                failed_wards.append(ward_name)
                
                # 連続して3つ以上の区でエラーが発生した場合は中断
                if len(failed_wards) >= 3:
                    logger.error(f"Too many failures ({len(failed_wards)} wards). Stopping data collection.")
                    logger.error(f"Failed wards: {', '.join(failed_wards)}")
                    break
        
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
            filename = self.data_dir / 'tokyo23_real_estate.csv'
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
                        'year': str(year),
                        'quarter': str(quarter),
                        'city': city_code,
                        'priceClassification': '01',  # 取引価格
                        'language': 'ja'
                    }
                    
                    # ヘッダーにAPIキーを設定
                    headers = self.session.headers.copy()
                    if self.api_key:
                        headers['Ocp-Apim-Subscription-Key'] = self.api_key
                    
                    # より詳細なログ
                    logger.debug(f"Fetching {ward_name} {year}Q{quarter}")
                    logger.debug(f"Request URL: {self.BASE_URL}")
                    logger.debug(f"Parameters: {params}")
                    
                    try:
                        response = self.session.get(
                            self.BASE_URL,
                            params=params,
                            headers=headers,
                            timeout=60,  # タイムアウトを延長
                            verify=True
                        )
                        
                        logger.debug(f"Response status: {response.status_code}")
                        
                        if response.status_code == 200:
                            # gzip圧縮されたレスポンスを処理
                            try:
                                response_data = response.json()
                                
                                # APIレスポンスの構造確認（status: OK, data: [...]）
                                if isinstance(response_data, dict) and 'status' in response_data:
                                    if response_data['status'] == 'OK' and 'data' in response_data:
                                        data = response_data['data']
                                        if isinstance(data, list) and len(data) > 0:
                                            # レコードに区名を追加
                                            for record in data:
                                                record['区'] = ward_name
                                            all_records.extend(data)
                                            logger.info(f"Found {len(data)} records for {ward_name} {year}Q{quarter}")
                                        else:
                                            # 空の配列 = その期間のデータなし
                                            logger.debug(f"No transaction data for {ward_name} {year}Q{quarter}")
                                    else:
                                        logger.warning(f"API returned non-OK status: {response_data.get('status')}")
                                        if 'message' in response_data:
                                            logger.warning(f"Message: {response_data['message']}")
                                else:
                                    logger.debug(f"Unexpected response format: {response_data}")
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON response: {e}")
                                logger.debug(f"Response content: {response.text[:500]}")
                        else:
                            logger.warning(f"API returned status {response.status_code} for {year}Q{quarter}")
                            if response.status_code == 404:
                                logger.error("404 Not Found - The API endpoint or parameters might be incorrect")
                                logger.error(f"URL: {response.url}")
                            elif response.status_code == 401:
                                logger.error("401 Unauthorized - API key is missing or invalid")
                            elif response.status_code == 403:
                                logger.error("403 Forbidden - API key does not have access to this resource")
                            logger.debug(f"Response headers: {dict(response.headers)}")
                            logger.debug(f"Response content: {response.text[:500]}")
                            
                    except requests.exceptions.ConnectionError as e:
                        logger.error(f"Connection error for {ward_name} {year}Q{quarter}: {e}")
                        logger.info("Retrying with alternative connection methods...")
                        
                        # 代替接続方法を試す
                        try:
                            # IPv4を強制
                            import urllib3
                            urllib3.util.connection.HAS_IPV6 = False
                            
                            response = requests.get(
                                self.BASE_URL,
                                params=params,
                                timeout=60,
                                headers=headers
                            )
                            
                            if response.status_code == 200:
                                response_data = response.json()
                                if isinstance(response_data, dict) and response_data.get('status') == 'OK':
                                    data = response_data.get('data', [])
                                    if isinstance(data, list) and len(data) > 0:
                                        for record in data:
                                            record['区'] = ward_name
                                        all_records.extend(data)
                                        logger.info(f"Alternative method succeeded: {len(data)} records")
                        except:
                            logger.error(f"All connection attempts failed for {ward_name} {year}Q{quarter}")
                        
                except requests.exceptions.Timeout:
                    logger.error(f"Timeout for {ward_name} {year}Q{quarter}")
                except Exception as e:
                    logger.error(f"Unexpected error for {ward_name} {year}Q{quarter}: {type(e).__name__}: {e}")
                    continue
                
                # レート制限対策
                time.sleep(0.5)
        
        if all_records:
            return pd.DataFrame(all_records)
        else:
            return pd.DataFrame()
    
    def _check_connection(self):
        """
        API接続をチェックしてDNS問題を検出
        """
        try:
            # DNSを解決できるか確認
            socket.gethostbyname('www.reinfolib.mlit.go.jp')
            logger.info("DNS resolution successful for www.reinfolib.mlit.go.jp")
            
            # 実際に接続できるか確認
            test_url = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
            headers = {'User-Agent': self.session.headers['User-Agent']}
            if self.api_key:
                headers['Ocp-Apim-Subscription-Key'] = self.api_key
            
            # テスト用のパラメータ（東京都千代田区、2023年第1四半期）
            test_params = {
                'year': '2023',
                'quarter': '1',
                'city': '13101',
                'priceClassification': '01',
                'language': 'ja'
            }
            
            response = self.session.get(test_url, params=test_params, headers=headers, timeout=10)
            logger.info(f"Connection test status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    test_data = response.json()
                    if isinstance(test_data, dict) and test_data.get('status') == 'OK':
                        data = test_data.get('data', [])
                        if isinstance(data, list):
                            logger.info(f"API test successful. Sample response has {len(data)} records")
                        else:
                            logger.info(f"API test successful but unexpected data format: {type(data)}")
                    else:
                        logger.warning(f"API test returned unexpected format: {test_data}")
                except:
                    logger.warning("Could not parse test response as JSON")
            
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed: {e}")
            logger.error("Please check your internet connection or try using alternative DNS servers")
            
            raise ConnectionError(f"DNS解決エラー: www.reinfolib.mlit.go.jp にアクセスできません。{e}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection test failed: {e}")
            logger.info("The API might be temporarily unavailable")
    
    def _clean_and_prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データのクリーニングと前処理
        """
        if df.empty:
            return df
        
        logger.info("Cleaning and preparing data...")
        
        # 必要な列を抽出（実際のAPIレスポンスに対応）
        # 既に日本語フィールド名で返されているため、マッピングは主に正規化用
        required_columns = [
            'Prefecture', 'Municipality', 'DistrictName', 'TradePrice', 'Area', 
            'TotalFloorArea', 'BuildingYear', 'Structure', 'Use', 'Purpose',
            'CityPlanning', 'Period', 'Type', 'FloorPlan', 'CoverageRatio', 
            'FloorAreaRatio', '区'
        ]
        
        # 利用可能な列のみを抽出
        available_columns = [col for col in required_columns if col in df.columns]
        if available_columns:
            df = df[available_columns]
        
        logger.info(f"Available columns: {list(df.columns)}")
        
        # 住宅・マンションのみを抽出（用途でフィルタリング）
        if 'Use' in df.columns:
            df = df[df['Use'].isin(['住宅', '共同住宅', '店舗兼住宅', '事務所兼住宅', ''])]
        
        # 数値データの変換
        numeric_columns = ['TradePrice', 'Area', 'TotalFloorArea', 'CoverageRatio', 'FloorAreaRatio']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 築年数の計算
        if 'BuildingYear' in df.columns:
            current_year = datetime.now().year
            df['建築年_西暦'] = df['BuildingYear'].apply(self._convert_to_western_year)
            df['築年数'] = current_year - df['建築年_西暦']
            df['築年数'] = df['築年数'].clip(lower=0)  # 負の値を0に
        
        # 面積データの正規化
        if 'Area' in df.columns:
            df['土地面積'] = df['Area']
        if 'TotalFloorArea' in df.columns:
            df['建物面積'] = df['TotalFloorArea']
        
        # 欠損値の処理
        if 'TradePrice' in df.columns:
            df = df.dropna(subset=['TradePrice'])
            
            # 異常値の除外（価格が極端に高い/低いものを除外）
            df['TradePrice_numeric'] = pd.to_numeric(df['TradePrice'], errors='coerce')
            df = df.dropna(subset=['TradePrice_numeric'])
            
            if len(df) > 0:
                price_lower = df['TradePrice_numeric'].quantile(0.01)
                price_upper = df['TradePrice_numeric'].quantile(0.99)
                df = df[(df['TradePrice_numeric'] >= price_lower) & 
                       (df['TradePrice_numeric'] <= price_upper)]
        
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
    
    # APIキーが必要な場合は環境変数MLIT_API_KEYに設定してください
    if not os.getenv('MLIT_API_KEY'):
        logger.warning("MLIT_API_KEY is not set. API access may be limited.")
    
    # ログレベルをINFOに設定（必要に応じてDEBUGに変更）
    logging.getLogger().setLevel(logging.INFO)
    
    # データ取得
    logger.info("Starting data collection for Tokyo 23 wards...")
    logger.info("Note: Data availability depends on MLIT API update schedule")
    
    # 現在の年と四半期を取得
    current_date = datetime.now()
    current_year = current_date.year
    current_quarter = (current_date.month - 1) // 3 + 1
    
    # データの遅延を考慮（通常1-2四半期遅れ）
    if current_quarter <= 2:
        to_year = current_year - 1
    else:
        to_year = current_year
    
    logger.info(f"Fetching data from 2022 to {to_year}")
    
    try:
        df = fetcher.fetch_all_tokyo23_data(from_year=2022, to_year=to_year, save_csv=True)
        
        if not df.empty:
            # 統計情報の表示
            stats = fetcher.get_summary_statistics(df)
            logger.info(f"Summary statistics: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        else:
            logger.error("ERROR: No data could be collected from the MLIT API.")
            logger.error("\n=== データ取得に失敗しました ===")
            logger.error("考えられる原因:")
            logger.error("1. APIキーが未設定または無効")
            logger.error("   → https://www.reinfolib.mlit.go.jp/ でAPIキーを取得してください")
            logger.error("2. インターネット接続の問題")
            logger.error("3. 国土交通省APIサーバーへのアクセス制限")
            logger.error("4. DNSの名前解決エラー (www.reinfolib.mlit.go.jp)")
            logger.error("\n対処方法:")
            logger.error("- APIキーを取得して環境変数 MLIT_API_KEY に設定")
            logger.error("  export MLIT_API_KEY='your-api-key-here'")
            logger.error("- DNS設定を確認してください (Google DNS: 8.8.8.8)")
            logger.error("- しばらく時間をおいてから再度実行してください")
            raise Exception("データ取得に失敗しました")
                
    except Exception as e:
        logger.error(f"\nERROR: {e}")
        logger.error("\n=== プログラムを終了します ===")
        raise