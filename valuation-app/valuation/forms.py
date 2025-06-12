from django import forms


class PropertyValuationForm(forms.Form):
    prefecture = forms.CharField(
        max_length=10,
        label='都道府県',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '例: 東京都'})
    )
    
    city = forms.CharField(
        max_length=50,
        label='市区町村',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '例: 渋谷区'})
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