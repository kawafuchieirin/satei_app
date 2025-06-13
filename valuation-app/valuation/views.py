import json
import requests
from django.shortcuts import render
from django.conf import settings
from django.contrib import messages
from .forms import PropertyValuationForm


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
                
                response = requests.post(
                    f"{settings.VALUATION_API_URL}/api/valuation",
                    json=valuation_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return render(request, 'valuation/result.html', {
                        'form': form,
                        'result': result,
                        'valuation_data': valuation_data
                    })
                else:
                    error_msg = f'API呼び出しに失敗しました。ステータス: {response.status_code}'
                    try:
                        error_detail = response.json()
                        error_msg += f' - {error_detail}'
                    except:
                        error_msg += f' - {response.text}'
                    messages.error(request, error_msg)
                    
            except requests.exceptions.RequestException as e:
                messages.error(request, f'APIとの通信でエラーが発生しました。: {str(e)}')
                
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
