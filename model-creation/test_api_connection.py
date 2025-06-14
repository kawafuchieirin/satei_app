#!/usr/bin/env python3
"""
国土交通省API接続テストスクリプト
APIの基本的な動作確認用
"""

import requests
import os
import json

def test_api_connection():
    """APIの接続テスト"""
    
    # API設定
    api_url = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
    api_key = os.getenv('MLIT_API_KEY')
    
    if not api_key:
        print("ERROR: MLIT_API_KEY environment variable is not set")
        print("Please set: export MLIT_API_KEY='your-api-key-here'")
        return
    
    # テストパラメータ（東京都千代田区、2023年第1四半期）
    params = {
        'year': '2023',
        'quarter': '1',
        'city': '13101',  # 千代田区
        'priceClassification': '01',
        'language': 'ja'
    }
    
    # ヘッダー
    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json'
    }
    
    print(f"Testing API connection...")
    print(f"URL: {api_url}")
    print(f"Parameters: {params}")
    print(f"API Key: {'Set' if api_key else 'Not set'}")
    print("-" * 50)
    
    try:
        # APIリクエスト
        response = requests.get(api_url, params=params, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"URL Called: {response.url}")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 50)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                if isinstance(data, list):
                    print(f"Success! Received {len(data)} records")
                    if len(data) > 0:
                        print("\nFirst record sample:")
                        print(json.dumps(data[0], indent=2, ensure_ascii=False))
                else:
                    print(f"Response type: {type(data)}")
                    print(f"Response data: {data}")
                    
            except json.JSONDecodeError:
                print("Failed to parse JSON response")
                print(f"Response text: {response.text[:500]}")
        else:
            print(f"Request failed with status {response.status_code}")
            print(f"Response text: {response.text[:500]}")
            
            if response.status_code == 404:
                print("\nPossible issues:")
                print("- Wrong API endpoint URL")
                print("- Invalid parameters")
            elif response.status_code == 401:
                print("\nPossible issues:")
                print("- Invalid or missing API key")
                print("- API key not properly set in header")
            elif response.status_code == 403:
                print("\nPossible issues:")
                print("- API key doesn't have access to this resource")
                print("- Account not activated")
                
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        print("\nPossible issues:")
        print("- No internet connection")
        print("- DNS resolution failed")
        print("- API server is down")

if __name__ == "__main__":
    test_api_connection()