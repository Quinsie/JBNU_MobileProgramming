# backend/source/scripts/weatherCollecter.py

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta

# 상대 경로 import 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.logger import log  # log 함수 추가

# API 설정
API_KEY = "ttHSb/Plt1ygMgUuYxHbMnRcDtDSvxIgpmoitKnjJG9ODIQ8/WjzBhsptYfc4/WF961ymr82GX4L/U0L28HuEA=="
ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

# 경로
COORDS_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_coords.json")
SAVE_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")

# 필요한 카테고리
CATEGORIES = ["PTY", "RN1", "T1H"]

# ±1 범위 내 대체 nx_ny 찾기
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


def collect_weather(nx, ny, coords, retry=2):
    now = datetime.now()
    base_dt = now.replace(minute=0, second=0, microsecond=0)
    try_count = 0
    tried_keys = set()
    origin_key = f"{nx}_{ny}"

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

                if not items:
                    log("weatherCollector", f"빈 응답 items → 재시도. (nx={nx}, ny={ny}) [시도 {try_count}]")
                    base_dt -= timedelta(minutes=30)
                    time.sleep(0.5)
                    continue

                result = {}
                for cat in CATEGORIES:
                    val = next((i["obsrValue"] for i in items if i["category"] == cat), None)
                    result[cat] = float(val) if val is not None and "." in str(val) else (int(val) if val is not None else None)

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

    # fallback 3회 실패 후 ±1 grid 탐색
    tried_keys.add(origin_key)
    alt_nx, alt_ny = get_alternate_nxny(nx, ny, coords, tried_keys)
    if alt_nx is not None:
        log("weatherCollector", f"±1 grid 대체 사용 시도: {alt_nx}_{alt_ny} (원래 {origin_key})")
        return collect_weather(alt_nx, alt_ny, coords, retry=retry)

    log("weatherCollector", f"최종 수집 실패: (nx={nx}, ny={ny})")
    return None

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
        log("weatherCollector", f"수집 중: {nx_ny}")

        weather = collect_weather(nx, ny, coords)
        if weather:
            collected[nx_ny] = weather
        else:
            log("weatherCollector", f"최종 대체 실패: {nx_ny}")

        time.sleep(1.0)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    log("weatherCollector", f"저장 완료: {save_path}")

if __name__ == "__main__":
    main()