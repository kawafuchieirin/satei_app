from django import template

register = template.Library()

@register.filter
def format_price_yen(value):
    """
    価格を「X,XXX万円」形式で表示するフィルター
    例: 39761889 → 3,976万円
    """
    try:
        price = float(value)
        # 万円単位に変換（四捨五入）
        man_yen = round(price / 10000)
        # カンマ区切りでフォーマット
        return f"{man_yen:,}万円"
    except (ValueError, TypeError):
        return value