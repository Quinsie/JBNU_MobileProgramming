# backend/source/scripts/fetchStops.py

import os
import json
import time
import requests

URL = "http://www.jeonjuits.go.kr/bis/selectBisRouteRsltList.do" # 노선별 정류장 API
SUBLIST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "subList"))
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "stops"))

def fetch_stops():
    os.makedirs(PATH, exist_ok=True)
    files = sorted(os.listdir(SUBLIST_PATH))

    for file in files:
        path = os.path.join(SUBLIST_PATH, file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        route_subs = data.get("resultList", [])
        for sub in route_subs:
            stdid = sub.get("BRT_STDID")
            if not stdid:
                continue

            payload = {
                "locale": "ko-kr",
                "routeId": stdid
            }

            try:
                res = requests.post(URL, data=payload, timeout=5)
                res.raise_for_status()
                result = res.json()

                save_path = os.path.join(PATH, f"{stdid}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                print(f"{stdid} 저장 완료")

            except Exception as error:
                print(f"{stdid} 저장 실패: {error}")

            time.sleep(0.5)  # 서버 과부하 방지

if __name__ == "__main__":
    fetch_stops()