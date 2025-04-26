# backend/source/tools/stdidToStops.py

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
STOPS_DIR = BASE_DIR / "data" / "raw" / "staticInfo" / "stops"
SAVE_PATH = BASE_DIR / "data" / "processed" / "stdid_to_stops.json"

def main():
    stdid_to_stops = {}

    # 모든 stops 디렉토리 순회
    for stop_file in STOPS_DIR.glob("*.json"):
        stdid = stop_file.stem  # 파일 이름이 STDID
        try:
            with open(stop_file, encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("resultList", []):
                    stop_ord = item.get("STOP_ORD")
                    node_id = item.get("NODE_ID")
                    if stop_ord is not None and node_id is not None:
                        key = f"{stdid}_{stop_ord}"
                        stdid_to_stops[key] = node_id
        except Exception as e:
            print(f"{stop_file.name} 읽기 실패: {e}")

    # 저장
    os.makedirs(SAVE_PATH.parent, exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(stdid_to_stops, f, indent=2, ensure_ascii=False)

    print(f"stdid_to_stops 저장 완료 → {SAVE_PATH}")

if __name__ == "__main__":
    main()