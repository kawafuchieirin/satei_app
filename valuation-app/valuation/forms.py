from django import forms


class PropertyValuationForm(forms.Form):
    # 東京都固定の選択肢
    TOKYO_WARDS = [
        ('千代田区', '千代田区'),
        ('中央区', '中央区'),
        ('港区', '港区'),
        ('新宿区', '新宿区'),
        ('文京区', '文京区'),
        ('台東区', '台東区'),
        ('墨田区', '墨田区'),
        ('江東区', '江東区'),
        ('品川区', '品川区'),
        ('目黒区', '目黒区'),
        ('大田区', '大田区'),
        ('世田谷区', '世田谷区'),
        ('渋谷区', '渋谷区'),
        ('中野区', '中野区'),
        ('杉並区', '杉並区'),
        ('豊島区', '豊島区'),
        ('北区', '北区'),
        ('荒川区', '荒川区'),
        ('板橋区', '板橋区'),
        ('練馬区', '練馬区'),
        ('足立区', '足立区'),
        ('葛飾区', '葛飾区'),
        ('江戸川区', '江戸川区'),
    ]
    
    prefecture = forms.CharField(
        max_length=10,
        label='都道府県',
        initial='東京都',
        widget=forms.TextInput(attrs={
            'class': 'form-control', 
            'value': '東京都',
            'readonly': True,
            'style': 'background-color: #f8f9fa;'
        })
    )
    
    city = forms.CharField(
        max_length=50,
        label='市区町村',
        widget=forms.TextInput(attrs={
            'class': 'form-control', 
            'placeholder': '例: 渋谷区',
            'list': 'tokyo-wards',
            'autocomplete': 'off'
        })
    )
    
    district = forms.CharField(
        max_length=50,
        label='地区',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '例: 渋谷1-1-1'})
    )
    
    land_area = forms.FloatField(
        label='土地面積（㎡）',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '例: 100.5'})
    )
    
    building_area = forms.FloatField(
        label='建物面積（㎡）',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '例: 80.0'})
    )
    
    building_age = forms.IntegerField(
        label='築年数',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '例: 10'})
    )