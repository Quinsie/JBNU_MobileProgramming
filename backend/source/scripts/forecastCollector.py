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
CATEGORIES = ["PTY", "RN1", "T1H"]

# 수집 함수 (fallback 추가)
def collect_forecast(nx, ny):
    now = datetime.now()
    base_dt = now - timedelta(days=1)
    base_times = ["2300", "2000", "1700"]  # fallback 순서

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

                    result[timestamp][category] = float(value) if "." in str(value) else int(value)

                # 여기! 성공한 base_date/base_time을 명시적으로 남긴다
                log("forecastCollector", f"[성공] {base_date} {base_time} 기준 예보 수집 완료 (nx={nx}, ny={ny})")
                return result

            except json.JSONDecodeError as e:
                log("forecastCollector", f"JSON 파싱 실패 (fallback 시도 중) {base_date} {base_time} (nx={nx}, ny={ny}): {e}")
        else:
            log("forecastCollector", f"요청 실패 (fallback 시도 중) {base_date} {base_time} (nx={nx}, ny={ny})")

        time.sleep(0.5)

    log("forecastCollector", f"[실패] 최종 수집 실패 (nx={nx}, ny={ny})")
    return None


def main():
    # 좌표 불러오기
    with open(COORDS_PATH, "r", encoding="utf-8") as f:
        coords = json.load(f)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d")  # 오늘 날짜
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{timestamp}.json")

    raw_collected = {}

    for nx_ny in coords:
        nx, ny = map(int, nx_ny.split("_"))
        log("forecastCollector", f"수집 중: {nx_ny} ({coords[nx_ny]['lat']}, {coords[nx_ny]['lng']})")

        forecast = collect_forecast(nx, ny)
        if forecast:
            raw_collected[nx_ny] = forecast
        else:
            raw_collected[nx_ny] = {}  # 실패시 빈 dict 기록

        time.sleep(1.0)

    # 데이터 재구성: timestamp 최상위
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