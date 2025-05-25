# backend/source/helpers/lookupAnalysis.py

import os
import json

def load_analysis_data(date_str, base_dir="data/processed/analysis/first_model"):
    path = os.path.join(base_dir, f"{date_str}.json")
    if not os.path.exists(path):
        print(f"[ERROR] 파일이 존재하지 않습니다: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def print_metrics(title, metrics_dict):
    print(f"\n==== {title} ====")
    for model_type, metrics in metrics_dict.items():
        print(f"\n[{model_type}]")
        for k, v in metrics.items():
            print(f"{k:>12}: {v}")

def main():
    date_str = input("날짜를 입력하세요 (YYYYMMDD): ").strip()
    data = load_analysis_data(date_str)
    if data is None:
        return

    category_map = {
        "total": ["overall"],
        "weekday": [f"weekday_{i}" for i in range(7)],
        "timegroup": [f"tg_{i}" for i in range(1, 9)],
        "wd_tg": [k for k in data if k.startswith("wdtg_")],
        "stdid": [k for k in data if k.startswith("route_")],
        "bus_number": [k for k in data if k.startswith("bus_")],
        "ord_ratio": [k for k in data if k.startswith("ord_")]
    }

    print("\n범주를 선택하세요:")
    for i, cat in enumerate(category_map):
        print(f"{i + 1}. {cat}")
    sel_idx = int(input("입력 번호: ")) - 1
    sel_key = list(category_map.keys())[sel_idx]

    options = category_map[sel_key]
    if sel_key == "total":
        print_metrics("overall", data["overall"])
        return

    print(f"\n선택 가능한 {sel_key} 하위 그룹:")
    for i, k in enumerate(options):
        print(f"{i + 1}. {k}")
    sub_idx = int(input("하위 그룹 번호 입력: ")) - 1
    key = options[sub_idx]

    if key in data:
        print_metrics(key, data[key])
    else:
        print(f"[ERROR] {key} 항목이 존재하지 않습니다.")

if __name__ == "__main__":
    main()