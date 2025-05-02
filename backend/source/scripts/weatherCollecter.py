# backend/source/scripts/weatherCollecter.py

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta

# 상대 경로 import 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.haversine import haversine_distance  # 거리 계산용
from source.utils.logger import log  # log 함수 추가

# API 설정
API_KEY = "ttHSb/Plt1ygMgUuYxHbMnRcDtDSvxIgpmoitKnjJG9ODIQ8/WjzBhsptYfc4/WF961ymr82GX4L/U0L28HuEA=="
ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

# 경로
COORDS_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_coords.json")
SAVE_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")

# 필요한 카테고리
CATEGORIES = ["PTY", "RN1", "T1H"]

def collect_weather(nx, ny, retry=2):
    now = datetime.now()
    base_dt = now.replace(minute=0, second=0, microsecond=0)
    try_count = 0

    while try_count <= retry:
        base_date = base_dt.strftime("%Y%m%d")
        base_time = base_dt.strftime("%H%M")

        params = {
            "serviceKey": API_KEY,
            "pageNo": 1,
            "numOfRows": 1000,
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": nx,
            "ny": ny
        }

        response = requests.get(ENDPOINT, params=params)
        try_count += 1

        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

                # 빈 응답일 경우 재시도
                if not items:
                    log("weatherCollector", f"빈 응답 items → 재시도. (nx={nx}, ny={ny}) [시도 {try_count}]")
                    base_dt -= timedelta(minutes=30)
                    time.sleep(0.5)
                    continue

                result = {}
                for cat in CATEGORIES:
                    val = next((i["obsrValue"] for i in items if i["category"] == cat), None)
                    if val is not None:
                        result[cat] = float(val) if "." in str(val) else int(val)
                    else:
                        result[cat] = None

                # 모든 값이 None일 경우 무효 처리
                if all(v is None for v in result.values()):
                    log("weatherCollector", f"모든 항목 None → 무효 응답 간주. 재시도. (nx={nx}, ny={ny}) [시도 {try_count}]")
                    base_dt -= timedelta(minutes=30)
                    time.sleep(0.5)
                    continue

                log("weatherCollector", f"대체 시각 사용: {base_time} (nx={nx}, ny={ny})")
                return result

            except json.JSONDecodeError as e:
                log("weatherCollector", f"JSON 파싱 실패 (nx={nx}, ny={ny}) [시도 {try_count}]: {e}")
        else:
            log("weatherCollector", f"응답 없음 or 오류 (nx={nx}, ny={ny}) [시도 {try_count}]: {base_time}")

        base_dt -= timedelta(minutes=30)
        time.sleep(0.5)

    log("weatherCollector", f"최종 수집 실패: (nx={nx}, ny={ny})")
    return None

def get_nearest_available(nx_ny, current_data, coords):
    if not current_data:
        return None

    target_coord = coords[nx_ny]
    target_pos = (target_coord["lat"], target_coord["lng"])

    min_dist = float("inf")
    nearest_value = None

    for other_key, val in current_data.items():
        if val is None or other_key == nx_ny:
            continue
        other_coord = coords[other_key]
        other_pos = (other_coord["lat"], other_coord["lng"])
        dist = haversine_distance(target_pos, other_pos)
        if dist < min_dist:
            min_dist = dist
            nearest_value = val

    if nearest_value:
        log("weatherCollector", f"거리 기반 대체 사용: {nx_ny} ← {min_dist:.2f}km 거리")
    return nearest_value

def main():
    # 좌표 불러오기
    with open(COORDS_PATH, "r", encoding="utf-8") as f:
        coords = json.load(f)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{timestamp}.json")

    collected = {}

    for nx_ny in coords:
        nx, ny = map(int, nx_ny.split("_"))
        log("weatherCollector", f"수집 중: {nx_ny} ({coords[nx_ny]['lat']}, {coords[nx_ny]['lng']})")

        weather = collect_weather(nx, ny)
        if weather:
            collected[nx_ny] = weather
        else:
            fallback = get_nearest_available(nx_ny, collected, coords)
            if fallback:
                collected[nx_ny] = fallback
            else:
                log("weatherCollector", f"최종 대체 실패: {nx_ny}")

        time.sleep(1.0)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    log("weatherCollector", f"저장 완료: {save_path}")

if __name__ == "__main__":
    
    main()