# source/utils.py

# 식품명 필터링 함수
def filter_product_names(texts):
    filtered = []
    
    for text in texts:
        """
        if len(text) < 2 or len(text) > 30:
            continue
        if any(w in text for w in ['결제', '합계', '카드', '포인트', '총액']):
            continue
        if any(c.isdigit() for c in text):
            continue
        """
        filtered.append(text)
    return list(set(filtered))  # 중복 제거