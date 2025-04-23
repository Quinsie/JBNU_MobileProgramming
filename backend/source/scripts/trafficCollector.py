# backend/source/scripts/trafficCollector.py
# 매 n초마다 API 요청으로 모든 도로 VERTEX에 대한 교통혼잡도를 가져오는 스크립트

import os
import sys
import json
import time
import requests
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.logger import log  # log 함수 추가

# 요청 설정
URL = "http://www.jeonjuits.go.kr/atms/selectTrafVrtxList.do"

PAYLOAD = {
    "minlat": 35.62887212891432,
    "maxlat": 36.010764597679966,
    "minlng": 126.81920706935077,
    "maxlng": 127.39975969071274,
    "link_lv": 2,
    "levl": 5
}

HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "Mozilla/5.0"
}

def fetch_traffic_data():
    response = requests.post(URL, data=PAYLOAD, headers=HEADERS)
    log("trafficCollector", f"응답 코드: {response.status_code}")  # 로그 추가

    if response.status_code != 200:
        log("trafficCollector", "요청 실패")
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
            "lat": lat,
            "lng": lng,
            "grade": grade
        })

    # SUB 필드 추가
    result = []
    for road_id, points in grouped.items():
        points.sort(key=lambda x: (x["lat"], x["lng"]))
        for sub_idx, point in enumerate(points):
            result.append({
                "id": point["id"],
                "sub": sub_idx,
                "grade": point["grade"]
            })

    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "traffic")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{timestamp}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    log("trafficCollector", f"저장 완료: {save_path}")  # 로그 추가

def main():
    data = fetch_traffic_data()
    if data:
        process_and_save(data)
    else:
        log("trafficCollector", "수신된 데이터 없음")

if __name__ == "__main__":
    main()