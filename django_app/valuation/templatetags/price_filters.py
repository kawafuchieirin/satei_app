from django import template

register = template.Library()

@register.filter
def format_price_yen(value):
    """
    価格を「X,XXX万円」形式で表示するフィルター
    注意: calculate_valuation()関数から返される値は既に万円単位
    """
    try:
        price = float(value)
        # 既に万円単位なので、カンマ区切りのみ追加
        man_yen = round(price)
        # カンマ区切りでフォーマット
        return f"{man_yen:,}万円"
    except (ValueError, TypeError):
        return value