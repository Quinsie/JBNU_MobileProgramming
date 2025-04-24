# /backend/source/tools/buildStopIndex.py
# 정류장을 기준으로 지나가는 노선STDID와 그 노선의 방향 (상/하행선)을 backend/data/processed/stop_to_routes/ 에 노선STDID.json별로 저장.

import os
import json
from collections import defaultdict

STOPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "stops"))
SUBLIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "subList"))
SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "stop_to_routes"))
os.makedirs(SAVE_DIR, exist_ok=True)

def load_sublist_mapping():
    stdid_to_direction = {}
    stdid_to_brtno = {}

    for file in os.listdir(SUBLIST_DIR):
        path = os.path.join(SUBLIST_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            sublist = json.load(f).get("resultList", [])
        for item in sublist:
            stdid = str(item.get("BRT_STDID"))
            direction = item.get("BRT_DIRECTION")
            brt_no = item.get("BRT_NO")
            stdid_to_direction[stdid] = direction
            stdid_to_brtno[stdid] = brt_no

    return stdid_to_direction, stdid_to_brtno

def build_stop_to_routes():
    stdid_to_direction, stdid_to_brtno = load_sublist_mapping()
    stop_to_routes = defaultdict(list)

    for file in os.listdir(STOPS_DIR):
        path = os.path.join(STOPS_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f).get("resultList", [])

        stdid = os.path.splitext(file)[0]
        direction = stdid_to_direction.get(stdid)
        brt_no = stdid_to_brtno.get(stdid)

        for stop in data:
            stop_id = str(stop.get("STOP_ID"))
            stop_to_routes[stop_id].append({
                "brt_no": brt_no,
                "stdid": stdid,
                "direction": int(direction) if direction else None
            })

    for stop_id, routes in stop_to_routes.items():
        save_path = os.path.join(SAVE_DIR, f"{stop_id}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(routes, f, ensure_ascii=False, indent=2)
        print(f"{stop_id} 저장 완료")

if __name__ == "__main__":
    build_stop_to_routes()