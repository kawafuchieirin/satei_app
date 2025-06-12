#!/usr/bin/env python3
"""
シンプルなモデル精度検証スクリプト
"""

import requests
import json
import time

# API URLの設定
API_BASE_URL = "http://localhost:3001"

def test_model_predictions():
    """
    モデルの予測精度をテスト
    """
    print("=" * 60)
    print("不動産査定モデル 簡易精度検証")
    print("=" * 60)
    
    # テストケース
    test_cases = [
        {
            "name": "東京都心の高級物件",
            "data": {
                "prefecture": "東京都",
                "city": "港区",
                "district": "赤坂",
                "land_area": 150.0,
                "building_area": 120.0,
                "building_age": 5
            },
            "expected_range": (80_000_000, 200_000_000)  # 8000万〜2億円
        },
        {
            "name": "東京都内の標準的な物件",
            "data": {
                "prefecture": "東京都",
                "city": "渋谷区",
                "district": "恵比寿",
                "land_area": 100.0,
                "building_area": 80.0,
                "building_age": 15
            },
            "expected_range": (40_000_000, 100_000_000)  # 4000万〜1億円
        },
        {
            "name": "築古物件",
            "data": {
                "prefecture": "東京都",
                "city": "世田谷区",
                "district": "三軒茶屋",
                "land_area": 80.0,
                "building_area": 60.0,
                "building_age": 35
            },
            "expected_range": (20_000_000, 60_000_000)  # 2000万〜6000万円
        },
        {
            "name": "小規模物件",
            "data": {
                "prefecture": "東京都",
                "city": "杉並区",
                "district": "高円寺",
                "land_area": 50.0,
                "building_area": 40.0,
                "building_age": 20
            },
            "expected_range": (15_000_000, 45_000_000)  # 1500万〜4500万円
        }
    ]
    
    results = []
    
    print(f"\n🔄 APIエンドポイント: {API_BASE_URL}/api/valuation")
    print(f"📝 テストケース数: {len(test_cases)}")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- テストケース {i}: {test_case['name']} ---")
        
        try:
            # API呼び出し
            response = requests.post(
                f"{API_BASE_URL}/api/valuation",
                json=test_case["data"],
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predicted_price = result["estimated_price"]
                confidence = result.get("confidence", "N/A")
                
                # 期待値範囲との比較
                min_expected, max_expected = test_case["expected_range"]
                is_in_range = min_expected <= predicted_price <= max_expected
                
                print(f"  📍 所在地: {test_case['data']['prefecture']} {test_case['data']['city']} {test_case['data']['district']}")
                print(f"  🏠 土地面積: {test_case['data']['land_area']}㎡")
                print(f"  🏢 建物面積: {test_case['data']['building_area']}㎡")
                print(f"  📅 築年数: {test_case['data']['building_age']}年")
                print(f"  💰 予測価格: ¥{predicted_price:,.0f}")
                print(f"  📊 信頼度: {confidence}%")
                print(f"  📈 期待範囲: ¥{min_expected:,.0f} - ¥{max_expected:,.0f}")
                print(f"  ✅ 範囲内: {'はい' if is_in_range else 'いいえ'}")
                
                if "factors" in result:
                    print(f"  🔍 査定要因:")
                    for factor in result["factors"][:3]:  # 上位3つ
                        print(f"     • {factor}")
                
                results.append({
                    "case": test_case["name"],
                    "predicted": predicted_price,
                    "confidence": confidence,
                    "in_range": is_in_range,
                    "deviation": calculate_deviation(predicted_price, test_case["expected_range"])
                })
                
            else:
                print(f"  ❌ APIエラー: {response.status_code}")
                print(f"  詳細: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"  ❌ 通信エラー: {e}")
        
        # 少し待機
        time.sleep(1)
    
    # 結果サマリー
    print_summary(results)


def calculate_deviation(predicted, expected_range):
    """
    期待値範囲からの偏差を計算
    """
    min_expected, max_expected = expected_range
    center = (min_expected + max_expected) / 2
    
    if min_expected <= predicted <= max_expected:
        return 0  # 範囲内
    elif predicted < min_expected:
        return (predicted - min_expected) / center * 100
    else:
        return (predicted - max_expected) / center * 100


def print_summary(results):
    """
    結果サマリーを表示
    """
    if not results:
        print("\n❌ 有効な結果がありません")
        return
    
    print("\n" + "=" * 60)
    print("📊 精度検証結果サマリー")
    print("=" * 60)
    
    # 統計計算
    total_cases = len(results)
    in_range_count = sum(1 for r in results if r["in_range"])
    accuracy_rate = (in_range_count / total_cases) * 100
    
    avg_confidence = sum(r["confidence"] for r in results if isinstance(r["confidence"], (int, float))) / total_cases
    
    print(f"\n📈 全体統計:")
    print(f"   テストケース数: {total_cases}")
    print(f"   期待範囲内の予測: {in_range_count}/{total_cases} ({accuracy_rate:.1f}%)")
    print(f"   平均信頼度: {avg_confidence:.1f}%")
    
    print(f"\n🎯 各テストケースの結果:")
    for result in results:
        status = "✅" if result["in_range"] else "❌"
        deviation_text = ""
        if result["deviation"] != 0:
            deviation_text = f" (偏差: {result['deviation']:+.1f}%)"
        
        print(f"   {status} {result['case']}: ¥{result['predicted']:,.0f}{deviation_text}")
    
    print(f"\n💡 評価:")
    if accuracy_rate >= 75:
        print("   🌟 優秀: モデルの予測精度は高く、実用的です")
    elif accuracy_rate >= 50:
        print("   👍 良好: モデルの予測精度は実用レベルです")
    elif accuracy_rate >= 25:
        print("   ⚠️  注意: モデルの予測精度に改善の余地があります")
    else:
        print("   🔴 問題: モデルの予測精度が低く、見直しが必要です")


def test_api_health():
    """
    API接続テスト
    """
    print("\n🔄 API接続テスト...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ API接続正常")
            return True
        else:
            print(f"❌ API接続エラー: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API接続失敗: {e}")
        return False


if __name__ == "__main__":
    # API接続確認
    if test_api_health():
        # モデル精度テスト実行
        test_model_predictions()
    else:
        print("\n❌ APIに接続できません。以下を確認してください:")
        print("   1. docker-compose up でサービスが起動しているか")
        print("   2. API_BASE_URL が正しいか")
        print("   3. ファイアウォール設定")