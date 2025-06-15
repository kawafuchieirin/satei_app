from django.urls import path
from . import views

app_name = 'valuation'

urlpatterns = [
    path('', views.index, name='index'),
    path('form/', views.valuation_form, name='form'),
    path('test-api/', views.test_api, name='test_api'),
    path('debug-api/', views.test_api, name='debug_api'),
    path('api/valuation/', views.api_valuation, name='api_valuation'),
]