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


def collect_weather_at(nx, ny, target_dt, retry=2):
    base_dt = target_dt.replace(second=0, microsecond=0)
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

                result = {}
                for cat in CATEGORIES:
                    val = next((i["obsrValue"] for i in items if i["category"] == cat), None)
                    result[cat] = float(val) if val is not None and "." in str(val) else (int(val) if val else None)

                log("weatherCollector", f"[백필] 수집 성공: {base_time} (nx={nx}, ny={ny})")
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
    min_dist = float("inf")
    nearest_value = None

    for other_key, val in current_data.items():
        if val is None or other_key == nx_ny:
            continue
        other_coord = coords[other_key]
        dist = haversine_distance(
            target_coord["lat"], target_coord["lng"],
            other_coord["lat"], other_coord["lng"]
        )
        if dist < min_dist:
            min_dist = dist
            nearest_value = val

    if nearest_value:
        log("weatherCollector", f"거리 기반 대체 사용: {nx_ny} ← {min_dist:.2f}km 거리")
    return nearest_value

def backfill_weather(start_str="20250421_0530", end_str="20250423_2330"):

    with open(COORDS_PATH, "r", encoding="utf-8") as f:
        coords = json.load(f)

    start_dt = datetime.strptime(start_str, "%Y%m%d_%H%M")
    end_dt = datetime.strptime(end_str, "%Y%m%d_%H%M")
    current = start_dt

    while current <= end_dt:
        if 1 <= current.hour <= 5 and current.minute == 0:
            current += timedelta(minutes=30)
            continue
        timestamp = current.strftime("%Y%m%d_%H%M")
        save_path = os.path.join(SAVE_DIR, f"{timestamp}.json")
        collected = {}

        log("weatherCollector", f"[백필] 수집 시작: {timestamp}")

        for nx_ny in coords:
            nx, ny = map(int, nx_ny.split("_"))
            weather = collect_weather_at(nx, ny, current)
            if weather:
                collected[nx_ny] = weather
            else:
                fallback = get_nearest_available(nx_ny, collected, coords)
                if fallback:
                    collected[nx_ny] = fallback
                else:
                    log("weatherCollector", f"[백필] 최종 대체 실패: {nx_ny}")
            time.sleep(0.5)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(collected, f, ensure_ascii=False, indent=2)

        log("weatherCollector", f"[백필] 저장 완료: {save_path}")
        current += timedelta(minutes=30)


if __name__ == "__main__":
    backfill_weather()