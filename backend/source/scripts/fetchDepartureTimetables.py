# backend/source/scripts/fetchDepartureTimetables.py

import os
import json
import time
import requests

URL = "http://www.jeonjuits.go.kr/bis/selectBisRouteTimeInfo.do" # 전주시 BIS 시간표 API (시점 시간표)
SUBLIST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "subList"))
SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "departure_timetables"))
os.makedirs(SAVE_PATH, exist_ok=True)

def fetch_timetables():
    # 모든 subList 파일 순회
    for filename in os.listdir(SUBLIST_PATH):
        filepath = os.path.join(SUBLIST_PATH, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            sublist_data = json.load(file)

        for item in sublist_data.get("resultList", []):
            stdid = item.get("BRT_STDID")
            brt_no = item.get("BRT_NO")
            direction = item.get("BRT_DIRECTION")

            payload = {
                "locale": "ko-kr",
                "routeId": stdid,
                "brtSubId": "0"
            }

            try:
                res = requests.post(URL, data=payload, timeout=5)
                res.raise_for_status()
                data = res.json()

                weekday = data.get("timeList", [])
                sat_nlist = data.get("result", {}).get("SAT_NLIST", "").strip()
                holi_nlist = data.get("result", {}).get("HOLI_NLIST", "").strip()

                # 감회 시간 제거: 실제 토/공휴일 운행 시간표 생성
                sat_removed = set(sat_nlist.split(", ")) if sat_nlist else set()
                holi_removed = set(holi_nlist.split(", ")) if holi_nlist else set()

                saturday = [t for t in weekday if t not in sat_removed]
                holiday = [t for t in weekday if t not in holi_removed]

                # 저장 구조
                result = {
                    "brt_no": brt_no,
                    "direction": direction,
                    "weekday": weekday,
                    "saturday": saturday,
                    "holiday": holiday
                }

                # 저장
                save_file = os.path.join(SAVE_PATH, f"{stdid}.json")
                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                print(f"저장 완료: STDID {stdid}")

            except Exception as e:
                print(f"저장 실패: STDID {stdid} | {e}")

            time.sleep(0.5)  # 과도한 요청 방지

if __name__ == "__main__":
    fetch_timetables()