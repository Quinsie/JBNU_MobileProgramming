# backend/source/utils/normalize.py

def normalize(value: float, min_val: float, max_val: float) -> float:
    return max(min((value - min_val) / (max_val - min_val), 1), 0)