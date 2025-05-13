# backend/source/tools/nxnyStopsGenerator.py

import os
import sys
import json

# BASE_DIR 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.convertToGrid import convert_to_grid

# 경로 설정
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")

nx_ny_stops = {}

for filename in os.listdir(STOP_DIR):
    if not filename.endswith(".json"):
        continue

    stdid = filename.replace(".json", "")
    file_path = os.path.join(STOP_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        stop_data = json.load(f)

    for stop in stop_data.get("resultList", []):
        stop_ord = stop.get("STOP_ORD")
        lat = stop.get("LAT")
        lng = stop.get("LNG")

        if None in (stop_ord, lat, lng):
            continue

        nx, ny = convert_to_grid(lat, lng)
        key = f"{stdid}_{stop_ord}"
        nx_ny_stops[key] = f"{nx}_{ny}"

# 저장
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(nx_ny_stops, f, ensure_ascii=False, indent=2)

print(f"nx_ny_stops.json 저장 완료: {SAVE_PATH} (총 {len(nx_ny_stops)}개)")