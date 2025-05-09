# backend/source/scripts/fetchTrafficVtxList.py
# 전주시 도로교통정보 API로부터 특정 도로의 특정 위치의 혼잡도를 알려주는 정보 중 혼잡도는 제외하고, 도로ID와 LAT/LONG을 backend/data/raw/staticInfo/traf_vtxlist.json에 저장.

import os
import json
import requests

URL = "http://www.jeonjuits.go.kr/atms/selectTrafVrtxList.do"
SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "traf_vtxlist.json"))

# 전주시 전체 영역 범위
payload = {
    "minlat": 35.62887212891432,
    "maxlat": 36.010764597679966,
    "minlng": 126.81920706935077,
    "maxlng": 127.39975969071274,
    "link_lv": 2,
    "levl": 5
}

def fetch_traffic_vtx():
    try:
        res = requests.post(URL, data=payload, timeout=10)
        res.raise_for_status()
        raw_data = res.json().get("resultList", [])

        # 필드 정제
        processed = [
            {
                "id": item["ID"],
                "lat": item["Y_CRDN"],
                "lng": item["X_CRDN"]
            }
            for item in raw_data
            if "ID" in item and "Y_CRDN" in item and "X_CRDN" in item
        ]

        # 정렬: ID → lat → lng
        processed.sort(key=lambda x: (x["id"], x["lat"], x["lng"]))

        # sub 필드 추가
        last_id = None
        sub_counter = 0
        for item in processed:
            if item["id"] != last_id:
                last_id = item["id"]
                sub_counter = 0
            item["sub"] = sub_counter
            sub_counter += 1

        # 저장
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        print("저장 완료:", SAVE_PATH)

    except Exception as e:
        print("오류 발생:", e)

if __name__ == "__main__":
    fetch_traffic_vtx()