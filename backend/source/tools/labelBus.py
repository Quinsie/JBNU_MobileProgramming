# backend/source/tools/labelBus.py
# 버스 번호 라벨링

import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROUTES_PATH = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "routes.json")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")

def build_bus_label():
    with open(ROUTES_PATH, "r", encoding="utf-8") as f:
        routes = json.load(f)  # list로 가정

    # 문자열 정렬 (숫자+하이픈 등 포함된 경우도 고려)
    sorted_routes = sorted(routes, key=lambda x: (int(x.split('-')[0]) if x.split('-')[0].isdigit() else float('inf'), x))
    
    label_dict = {route: str(i+1) for i, route in enumerate(sorted_routes)}

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_bus_label()