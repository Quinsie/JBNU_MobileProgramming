# /backend/source/scripts/buildStopIndex.py

import os
import json

# 경로 설정
STOPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "stops"))
SUBLIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "subList"))
SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "stop_to_routes.json"))

def build_index():
    stop_index = dict()

    # 모든 노선번호(subList) 순회
    for filename in os.listdir(SUBLIST_DIR):
        if not filename.endswith(".json"):
            continue

        bus_no = filename.replace(".json", "")
        filepath = os.path.join(SUBLIST_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            sublist = json.load(f)

        for sub in sublist.get("resultList", []):
            stdid = str(sub.get("BRT_STDID"))
            direction = int(sub.get("BRT_DIRECTION", 0))

            # 해당 stdid 정류장 목록 불러오기
            stop_file = os.path.join(STOPS_DIR, f"{stdid}.json")
            if not os.path.exists(stop_file):
                continue

            with open(stop_file, "r", encoding="utf-8") as f:
                stop_data = json.load(f)

            for stop in stop_data.get("resultList", []):
                stop_id = str(stop.get("STOP_ID"))

                stop_index.setdefault(stop_id, []).append({
                    "brt_no": bus_no,
                    "stdid": stdid,
                    "direction": direction
                })

    # 저장
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(stop_index, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    build_index()