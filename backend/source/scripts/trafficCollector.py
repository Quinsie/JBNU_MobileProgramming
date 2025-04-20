# backend/source/scripts/trafficCollector.py
# 매 n초마다 API 요청으로 모든 도로 VERTEX에 대한 교통혼잡도를 가져오는 스크립트트

import os
import json
import time
import requests
from datetime import datetime

# 요청 설정
URL = "http://www.jeonjuits.go.kr/atms/selectTrafVrtxList.do"

PAYLOAD = {
    "minlat": 35.76658320160097,
    "maxlat": 35.881663807032936,
    "minlng": 127.0341816493253,
    "maxlng": 127.26192077595161,
    "link_lv": 3,
    "levl": 2
}

HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "Mozilla/5.0"
}

def fetch_traffic_data():
    response = requests.post(URL, data=PAYLOAD, headers=HEADERS)
    print(response.status_code, flush=True)

    if response.status_code != 200:
        print("요청 실패", flush=True)
        return []

    data = response.json().get("resultList", [])
    return data

def process_and_save(data):
    # ID 기준으로 그룹핑
    grouped = {}
    for item in data:
        road_id = item["ID"]
        lat = item["Y_CRDN"]
        lng = item["X_CRDN"]
        grade = item["GRADE"]

        if road_id not in grouped:
            grouped[road_id] = []

        grouped[road_id].append({
            "id": road_id,
            "lat": lat,   # 정렬용으로만 넣음
            "lng": lng,
            "grade": grade
        })

    # SUB 필드 추가
    result = []
    for road_id, points in grouped.items():
        points.sort(key=lambda x: (x["lat"], x["lng"]))  # 좌표 정렬
        for sub_idx, point in enumerate(points):
            result.append({
                "id": point["id"],
                "sub": sub_idx,
                "grade": point["grade"]
            })

    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("backend", "data", "raw", "dynamicInfo", "traffic")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestamp}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {save_path}", flush=True)

def main():
    data = fetch_traffic_data()
    if data:
        process_and_save(data)

if __name__ == "__main__":
    main()