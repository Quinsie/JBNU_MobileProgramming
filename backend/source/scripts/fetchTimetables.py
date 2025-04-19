# backend/source/scripts/fetchTimetable.py

import os
import json
import time
import requests

# 경로 설정
URL = "http://www.jeonjuits.go.kr/bis/selectTimeTableInformation.do"
SUBLIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "subList"))
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "timetable"))

def fetch_timetables():
    os.makedirs(PATH, exist_ok=True)
    files = sorted(os.listdir(SUBLIST_DIR))

    for file_name in files:
        brt_no = file_name.replace(".json", "")
        path = os.path.join(SUBLIST_DIR, file_name)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sub_list = data.get("resultList", [])
        for sub in sub_list:
            stdid = sub.get("BRT_STDID")
            if not stdid:
                continue

            payload = {
                "locale": "ko-kr",
                "routeId": stdid,
                "brtSubId": "0"
            }

            try:
                res = requests.post(URL, data=payload, timeout=5)
                res.raise_for_status()
                result = res.json()

                save_path = os.path.join(PATH, f"{stdid}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                print(f"시간표 저장 완료: {brt_no} - STDID {stdid}")

            except Exception as e:
                print(f"저장 실패: {brt_no} - STDID {stdid} | {e}")

            time.sleep(0.5)

if __name__ == "__main__":
    fetch_timetables()