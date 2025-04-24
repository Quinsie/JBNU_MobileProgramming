# backend/source/tools/generateStopsGrid.py
# 정류장 좌표 기반으로 nx_ny 매핑 생성 → nx_ny_stops.json 저장

import os
import sys
import json
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.convertToGrid import convert_to_grid

STOPS_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")

nxny_map = {}  # {"STDID_ORD": "nx_ny"}

def generate():
    for fname in tqdm(os.listdir(STOPS_DIR)):
        if not fname.endswith(".json"):
            continue

        stdid = fname.replace(".json", "")
        path = os.path.join(STOPS_DIR, fname)

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            for stop in data.get("resultList", []):
                ord = stop.get("STOP_ORD")
                lat = stop.get("LAT")
                lng = stop.get("LNG")
                if lat is None or lng is None:
                    continue

                x, y = convert_to_grid(lat, lng)
                key = f"{stdid}_{ord}"
                nxny_map[key] = f"{x}_{y}"

        except Exception as e:
            print(f"[오류] {fname} → {e}")

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(nxny_map, f, ensure_ascii=False, indent=2)

    print(f"✅ 총 {len(nxny_map)}개 정류장에 대한 nx_ny 매핑 완료 → {SAVE_PATH}")

if __name__ == "__main__":
    generate()
