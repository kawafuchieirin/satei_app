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
                    'district': form.cleaned_data['district'],
                    'land_area': form.cleaned_data['land_area'],
                    'building_area': form.cleaned_data['building_area'],
                    'building_age': form.cleaned_data['building_age']
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
                    messages.error(request, 'API呼び出しに失敗しました。')
                    
            except requests.exceptions.RequestException as e:
                messages.error(request, 'APIとの通信でエラーが発生しました。')
                
    else:
        form = PropertyValuationForm()
    
    return render(request, 'valuation/form.html', {'form': form})


def index(request):
    return render(request, 'valuation/index.html')
