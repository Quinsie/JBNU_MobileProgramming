# backend/source/scripts/forecastCollector.py

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta

# 상대 경로 import 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.haversine import haversine_distance  # 거리 계산용 (혹시 몰라 추가)
from source.utils.logger import log  # log 함수 추가

# API 설정
API_KEY = "ttHSb/Plt1ygMgUuYxHbMnRcDtDSvxIgpmoitKnjJG9ODIQ8/WjzBhsptYfc4/WF961ymr82GX4L/U0L28HuEA=="
ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

# 경로
COORDS_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_coords.json")
SAVE_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")

# 필요한 카테고리
CATEGORIES = ["PTY", "PCP", "TMP"]

def get_alternate_nxny(nx, ny, coords, tried_set):
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            alt_nx, alt_ny = nx + dx, ny + dy
            alt_key = f"{alt_nx}_{alt_ny}"
            if alt_key in coords and alt_key not in tried_set:
                return alt_nx, alt_ny
    return None, None

# 수집 함수 (fallback 추가)
def collect_forecast(nx, ny, coords):
    now = datetime.now()
    base_dt = now - timedelta(days=1)
    base_times = ["2300", "2000", "1700"]  # fallback 순서
    tried_keys = set()
    origin_key = f"{nx}_{ny}"

    for base_time in base_times:
        base_date = base_dt.strftime("%Y%m%d")

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

        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

                if not items:
                    log("forecastCollector", f"빈 데이터 (fallback 시도 중): {base_date} {base_time} (nx={nx}, ny={ny})")
                    time.sleep(0.5)
                    continue

                result = {}
                for item in items:
                    fcst_date = item.get("fcstDate")
                    fcst_time = item.get("fcstTime")
                    category = item.get("category")
                    value = item.get("fcstValue")

                    if category not in CATEGORIES:
                        continue

                    timestamp = f"{fcst_date}_{fcst_time}"

                    if timestamp not in result:
                        result[timestamp] = {}

                    result[timestamp][category] = float(value) if str(value).replace('.', '', 1).isdigit() else 0.0

                if not result:
                    log("forecastCollector", f"유효한 카테고리 없음 → fallback 계속 (nx={nx}, ny={ny})")
                    time.sleep(0.5)
                    continue

                log("forecastCollector", f"[성공] {base_date} {base_time} 기준 예보 수집 완료 (nx={nx}, ny={ny})")
                return result

            except json.JSONDecodeError as e:
                log("forecastCollector", f"JSON 파싱 실패 (fallback 시도 중) {base_date} {base_time} (nx={nx}, ny={ny}): {e}")
        else:
            log("forecastCollector", f"요청 실패 (fallback 시도 중) {base_date} {base_time} (nx={nx}, ny={ny})")

        time.sleep(0.5)

    # fallback 실패 시 ±1 grid 대체 시도
    tried_keys.add(origin_key)
    alt_nx, alt_ny = get_alternate_nxny(nx, ny, coords, tried_keys)
    if alt_nx is not None:
        log("forecastCollector", f"±1 grid 대체 사용 시도: {alt_nx}_{alt_ny} (원래 {origin_key})")
        return collect_forecast(alt_nx, alt_ny, coords)

    log("forecastCollector", f"[실패] 최종 수집 실패: (nx={nx}, ny={ny})")
    return None

def main():
    # 좌표 불러오기
    with open(COORDS_PATH, "r", encoding="utf-8") as f:
        coords = json.load(f)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d")
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{timestamp}.json")

    raw_collected = {}

    for nx_ny in coords:
        nx, ny = map(int, nx_ny.split("_"))
        log("forecastCollector", f"수집 중: {nx_ny}")

        forecast = collect_forecast(nx, ny, coords)
        if forecast:
            raw_collected[nx_ny] = forecast
        else:
            raw_collected[nx_ny] = {}

        time.sleep(1.0)

    # timestamp 기준으로 재구성
    reorganized = {}
    for nx_ny, forecasts in raw_collected.items():
        for ts, values in forecasts.items():
            if ts not in reorganized:
                reorganized[ts] = {}
            reorganized[ts][nx_ny] = values

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(reorganized, f, ensure_ascii=False, indent=2)

    log("forecastCollector", f"저장 완료: {save_path}")

if __name__ == "__main__":
    main()