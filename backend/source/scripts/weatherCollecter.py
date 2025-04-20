# backend/source/scripts/weatherCollecter.py

import os
import sys
import json
import time
import requests
from datetime import datetime

# 상대 경로 import 설정
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from convertToGrid import convert_to_grid  # 혹시 몰라서 유지

# API 설정
API_KEY = "ttHSb/Plt1ygMgUuYxHbMnRcDtDSvxIgpmoitKnjJG9ODIQ8/WjzBhsptYfc4/WF961ymr82GX4L/U0L28HuEA=="
ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

# 경로
COORDS_PATH = os.path.join("backend", "data", "processed", "nx_ny_coords.json")
SAVE_DIR = os.path.join("backend", "data", "raw", "dynamicInfo", "weather")

# 필요한 카테고리
CATEGORIES = ["PTY", "RN1", "T1H"]

def collect_weather(nx, ny):
    now = datetime.now()
    base_date = now.strftime("%Y%m%d")
    base_time = now.strftime("%H%M")

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

            result = {}
            for cat in CATEGORIES:
                val = next((i["obsrValue"] for i in items if i["category"] == cat), None)
                if val is not None:
                    result[cat] = float(val) if "." in str(val) else int(val)
                else:
                    result[cat] = None

            return result

        except Exception as e:
            print(f"JSON 파싱 실패 (nx={nx}, ny={ny}): {e}")
            return None
    else:
        print(f"응답 오류 (nx={nx}, ny={ny}): {response.status_code}")
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
        print(f"수집 중: {nx_ny} ({coords[nx_ny]['lat']}, {coords[nx_ny]['lng']})")

        weather = collect_weather(nx, ny)
        if weather:
            collected[nx_ny] = weather
        else:
            print(f"수집 실패: {nx_ny}")

        time.sleep(1.0)  # 과도한 요청 방지

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {save_path}")

if __name__ == "__main__":
    main()