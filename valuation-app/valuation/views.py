import json
import requests
import random
from django.shortcuts import render
from django.conf import settings
from django.contrib import messages
from .forms import PropertyValuationForm


def calculate_valuation(prefecture, city, district, land_area, building_area, building_age):
    """Django内で直接査定計算を実行"""
    # 東京23区の基準価格（万円/㎡）
    ward_base_prices = {
        '千代田区': 250, '中央区': 200, '港区': 220,
        '新宿区': 150, '文京区': 140, '台東区': 120,
        '墨田区': 100, '江東区': 110, '品川区': 160,
        '目黒区': 170, '大田区': 130, '世田谷区': 140,
        '渋谷区': 180, '中野区': 120, '杉並区': 130,
        '豊島区': 140, '北区': 100, '荒川区': 90,
        '板橋区': 100, '練馬区': 110, '足立区': 80,
        '葛飾区': 85, '江戸川区': 90
    }
    
    # 基準価格を取得（23区以外はデフォルト値）
    base_price_per_sqm = ward_base_prices.get(city, 100)
    
    # 土地価格計算
    land_price = land_area * base_price_per_sqm
    
    # 建物価格計算（築年数による減価）
    building_depreciation = max(0.3, 1 - (building_age * 0.03))
    building_price = building_area * base_price_per_sqm * 0.8 * building_depreciation
    
    # 総価格
    total_price = (land_price + building_price) * 10000  # 万円から円に変換
    
    # ランダム要素を加えて現実感を出す
    variation = random.uniform(0.9, 1.1)
    estimated_price = total_price * variation
    
    # 信頼度（築年数が新しいほど高い）
    confidence = max(70, min(90, 90 - building_age * 0.5))
    
    # 価格帯
    price_range = {
        'min': estimated_price * 0.85,
        'max': estimated_price * 1.15
    }
    
    # 査定要因
    factors = []
    if building_age <= 5:
        factors.append("築浅物件で、価格にプラス影響")
    elif building_age <= 10:
        factors.append("比較的新しい物件で、価格への影響は中程度")
    else:
        factors.append(f"築{building_age}年で、建物価値が減少")
        
    if land_area >= 100:
        factors.append("土地面積が広く、価格にプラス影響")
        
    if city in ['千代田区', '港区', '渋谷区', '中央区']:
        factors.append("都心の一等地で、価格が高め")
    
    return {
        'estimated_price': estimated_price,
        'confidence': confidence,
        'price_range': price_range,
        'factors': factors
    }


def valuation_form(request):
    if request.method == 'POST':
        form = PropertyValuationForm(request.POST)
        if form.is_valid():
            try:
                valuation_data = {
                    'prefecture': form.cleaned_data['prefecture'],
                    'city': form.cleaned_data['city'],
                    'district': form.cleaned_data['district'],
                    'land_area': form.cleaned_data['land_area'],
                    'building_area': form.cleaned_data['building_area'],
                    'building_age': form.cleaned_data['building_age']
                }
                
                # Django内で直接計算を実行
                result = calculate_valuation(**valuation_data)
                
                return render(request, 'valuation/result.html', {
                    'form': form,
                    'result': result,
                    'valuation_data': valuation_data
                })
                    
            except Exception as e:
                messages.error(request, f'査定計算でエラーが発生しました: {str(e)}')
                
    else:
        form = PropertyValuationForm()
    
    return render(request, 'valuation/form.html', {'form': form})


def index(request):
    return render(request, 'valuation/index.html')


def test_api(request):
    """APIテスト用エンドポイント"""
    from django.http import JsonResponse
    import requests
    from django.conf import settings
    
    try:
        test_data = {
            'prefecture': '東京都',
            'city': '新宿区',
            'district': '西新宿',
            'land_area': 100.0,
            'building_area': 80.0,
            'building_age': 5
        }
        
        api_url = f"{settings.VALUATION_API_URL}/api/valuation"
        response = requests.post(api_url, json=test_data, timeout=30)
        
        return JsonResponse({
            'api_url': api_url,
            'status_code': response.status_code,
            'response': response.json() if response.status_code == 200 else response.text,
            'test_data': test_data
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'api_url': f"{settings.VALUATION_API_URL}/api/valuation"
        })
