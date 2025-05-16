# /backend/source/tools/buildStopIndex.py
# 정류장을 기준으로 지나가는 노선STDID와 그 노선의 방향 (상/하행선)을 backend/data/processed/stop_to_routes/ 에 노선STDID.json별로 저장.

import os
import json
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STOPS_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "stop_to_routes.json")

def build_stop_to_routes():
    stop_to_routes = defaultdict(list)

    for file in os.listdir(STOPS_DIR):
        stdid = os.path.splitext(file)[0]
        path = os.path.join(STOPS_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            stops = json.load(f).get("resultList", [])

        for stop in stops:
            stop_id = str(stop.get("STOP_ID"))
            stop_ord = stop.get("STOP_ORD")

            if stop_id and stop_ord is not None:
                stop_to_routes[stop_id].append({
                    "stdid": stdid,
                    "ord": int(stop_ord)
                })

    # 통합 저장
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(stop_to_routes, f, ensure_ascii=False, indent=2)

    print(f"총 {len(stop_to_routes)}개 정류장에 대한 stop_to_routes 저장 완료")

if __name__ == "__main__":
    build_stop_to_routes()