import json
import requests
from django.shortcuts import render
from django.conf import settings
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .forms import PropertyValuationForm


def calculate_valuation(data):
    """内部査定処理関数"""
    # 基本価格データ（東京23区）
    base_price_per_sqm = {
        '千代田区': 1200000, '中央区': 1000000, '港区': 1100000,
        '新宿区': 800000, '文京区': 700000, '台東区': 600000,
        '墨田区': 550000, '江東区': 600000, '品川区': 750000,
        '目黒区': 800000, '大田区': 650000, '世田谷区': 700000,
        '渋谷区': 900000, '中野区': 650000, '杉並区': 600000,
        '豊島区': 650000, '北区': 550000, '荒川区': 500000,
        '板橋区': 550000, '練馬区': 500000, '足立区': 450000,
        '葛飾区': 450000, '江戸川区': 500000
    }
    
    # デフォルト価格（東京都平均）
    price_per_sqm = base_price_per_sqm.get(data['city'], 600000)
    
    # 築年数による減価計算
    age_factor = max(0.3, 1.0 - (data['building_age'] * 0.03))
    
    # 建物面積による計算
    building_value = data['building_area'] * price_per_sqm * age_factor
    
    # 土地面積による計算
    land_value = data['land_area'] * price_per_sqm * 0.8
    
    # 総査定額
    estimated_price = (building_value + land_value) / 10000  # 万円単位
    
    # 信頼度計算（簡易版）
    confidence = min(95, max(60, 85 - (data['building_age'] * 0.5)))
    
    # 価格レンジ
    price_range = {
        'min': estimated_price * 0.85,
        'max': estimated_price * 1.15
    }
    
    # 査定要因
    factors = [
        f"{data['city']}の基準価格: {price_per_sqm:,}円/㎡",
        f"築{data['building_age']}年による減価率: {int((1-age_factor)*100)}%",
        f"建物面積: {data['building_area']}㎡",
        f"土地面積: {data['land_area']}㎡"
    ]
    
    if data['building_age'] <= 5:
        factors.append("築浅物件でプラス評価")
    elif data['building_age'] >= 20:
        factors.append("築古物件でマイナス評価")
    
    return {
        'estimated_price': round(estimated_price, 2),
        'confidence': round(confidence, 1),
        'price_range': {
            'min': round(price_range['min'], 2),
            'max': round(price_range['max'], 2)
        },
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
                    'district': form.cleaned_data['district'],  # FastAPI expects 'district'
                    'land_area': form.cleaned_data['land_area'],
                    'building_area': form.cleaned_data['building_area'],
                    'building_age': form.cleaned_data['building_age']  # FastAPI expects 'building_age'
                }
                
                # 内部査定処理を直接実行
                result = calculate_valuation(valuation_data)
                
                # デバッグ用: 結果をログ出力
                print(f"DEBUG - valuation_data: {valuation_data}")
                print(f"DEBUG - result: {result}")
                
                return render(request, 'valuation/result.html', {
                    'form': form,
                    'result': result,
                    'valuation_data': valuation_data
                })
                    
            except Exception as e:
                messages.error(request, f'査定処理中にエラーが発生しました: {str(e)}')
                
    else:
        form = PropertyValuationForm()
    
    return render(request, 'valuation/form.html', {'form': form})


def index(request):
    return render(request, 'valuation/index.html')


@csrf_exempt
@require_http_methods(["POST"])
def api_valuation(request):
    """ML不動産査定APIエンドポイント"""
    try:
        data = json.loads(request.body)
        
        # 入力バリデーション
        required_fields = ['prefecture', 'city', 'land_area', 'building_area', 'building_age']
        for field in required_fields:
            if field not in data:
                return JsonResponse({
                    'error': f'必須フィールド {field} が不足しています'
                }, status=400)
        
        # 基本的なバリデーション
        if data['land_area'] <= 0 or data['building_area'] <= 0:
            return JsonResponse({
                'error': '土地面積と建物面積は正の数値である必要があります'
            }, status=400)
        
        if data['building_age'] < 0:
            return JsonResponse({
                'error': '築年数は0以上である必要があります'
            }, status=400)
        
        # 簡易査定ロジック（東京23区の基本ロジック）
        base_price_per_sqm = {
            '千代田区': 1200000, '中央区': 1000000, '港区': 1100000,
            '新宿区': 800000, '文京区': 700000, '台東区': 600000,
            '墨田区': 550000, '江東区': 600000, '品川区': 750000,
            '目黒区': 800000, '大田区': 650000, '世田谷区': 700000,
            '渋谷区': 900000, '中野区': 650000, '杉並区': 600000,
            '豊島区': 650000, '北区': 550000, '荒川区': 500000,
            '板橋区': 550000, '練馬区': 500000, '足立区': 450000,
            '葛飾区': 450000, '江戸川区': 500000
        }
        
        # デフォルト価格（東京都平均）
        price_per_sqm = base_price_per_sqm.get(data['city'], 600000)
        
        # 築年数による減価計算
        age_factor = max(0.3, 1.0 - (data['building_age'] * 0.03))
        
        # 建物面積による計算
        building_value = data['building_area'] * price_per_sqm * age_factor
        
        # 土地面積による計算
        land_value = data['land_area'] * price_per_sqm * 0.8
        
        # 総査定額
        estimated_price = (building_value + land_value) / 10000  # 万円単位
        
        # 信頼度計算（簡易版）
        confidence = min(95, max(60, 85 - (data['building_age'] * 0.5)))
        
        # 価格レンジ
        price_range = {
            'min': estimated_price * 0.85,
            'max': estimated_price * 1.15
        }
        
        # 査定要因
        factors = [
            f"{data['city']}の基準価格: {price_per_sqm:,}円/㎡",
            f"築{data['building_age']}年による減価率: {int((1-age_factor)*100)}%",
            f"建物面積: {data['building_area']}㎡",
            f"土地面積: {data['land_area']}㎡"
        ]
        
        if data['building_age'] <= 5:
            factors.append("築浅物件でプラス評価")
        elif data['building_age'] >= 20:
            factors.append("築古物件でマイナス評価")
        
        return JsonResponse({
            'estimated_price': round(estimated_price, 2),
            'confidence': round(confidence, 1),
            'price_range': {
                'min': round(price_range['min'], 2),
                'max': round(price_range['max'], 2)
            },
            'factors': factors
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': '無効なJSONデータです'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'査定処理中にエラーが発生しました: {str(e)}'
        }, status=500)


def test_api(request):
    """APIテスト用エンドポイント - 内部処理をテスト"""
    try:
        test_data = {
            'prefecture': '東京都',
            'city': '新宿区',
            'district': '西新宿',
            'land_area': 100.0,
            'building_area': 80.0,
            'building_age': 5
        }
        
        # 内部計算関数を直接テスト
        result = calculate_valuation(test_data)
        
        # 価格フィルター結果もテスト
        from .templatetags.price_filters import format_price_yen
        formatted_price = format_price_yen(result['estimated_price'])
        
        return JsonResponse({
            'test_data': test_data,
            'calculation_result': result,
            'formatted_price': formatted_price,
            'debug_info': {
                'raw_price': result['estimated_price'],
                'formatted': formatted_price
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'traceback': str(e)
        })
