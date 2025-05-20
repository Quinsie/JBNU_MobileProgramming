# backend/source/tools/labelStops.py
# 정류장 ID 라벨링

import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STDID_TO_STOPS_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_to_stops.json")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")

def build_stop_label():
    with open(STDID_TO_STOPS_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)  # dict

    # value만 모아서 중복 제거
    stop_ids = sorted(set(raw_data.values()))
    label_dict = {str(stop_id): str(i + 1) for i, stop_id in enumerate(stop_ids)}

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(label_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    build_stop_label()