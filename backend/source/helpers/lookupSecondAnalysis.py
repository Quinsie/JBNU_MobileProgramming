# backend/source/helpers/lookupSecondAnalysis.py

import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load_analysis_data(date_str, base_dir=os.path.join(BASE_DIR, "data", "processed", "analysis", "second_model")):
    path = os.path.join(base_dir, f"{date_str}.json")
    if not os.path.exists(path):
        print(f"[ERROR] 파일이 존재하지 않습니다: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def print_metrics(title, metrics_dict):
    print(f"\n==== {title.upper()} ====\n")

    ord_keys = list(metrics_dict.keys())
    metric_names = list(metrics_dict[ord_keys[0]].keys())

    header = f"{'METRIC':>15}" + "".join([f"{ord_key:>12}" for ord_key in ord_keys])
    print(header)
    print("-" * len(header))

    for metric in metric_names:
        row = f"{metric.upper():>15}"
        for ord_key in ord_keys:
            val = metrics_dict[ord_key][metric]
            if isinstance(val, float):
                row += f"{val:>12,.3f}"
            else:
                row += f"{str(val):>12}"
        print(row)

def main():
    date_str = input("날짜를 입력하세요 (YYYYMMDD): ").strip()
    data = load_analysis_data(date_str)
    if data is None:
        return

    category_map = {
        "total": ["overall"],
        "weekday": [f"weekday_{i}" for i in range(1, 4)],  # 1~3 사용
        "timegroup": [f"tg_{i}" for i in range(8)],
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
        print_metrics("overall", data["overall"]["overall"])
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