# backend/source/tools/mapLastStop.py

import os
import sys
import json

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "last_stop.json")

def extract_last_stop_ord(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
        stop_list = data.get("resultList", [])
        if not stop_list:
            return None
        return stop_list[-1].get("STOP_ORD")

if __name__ == "__main__":
    result = {}
    for fname in os.listdir(STOP_DIR):
        if not fname.endswith(".json"):
            continue
        stdid = fname.replace(".json", "")
        path = os.path.join(STOP_DIR, fname)
        last_ord = extract_last_stop_ord(path)
        if last_ord is not None:
            result[stdid] = last_ord

    with open(SAVE_PATH, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {SAVE_PATH}")