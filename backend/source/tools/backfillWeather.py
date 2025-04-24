import os
import json
import time
import requests
from datetime import datetime, timedelta

# 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COORDS_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_coords.json")
SAVE_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")

API_KEY = "ttHSb/Plt1ygMgUuYxHbMnRcDtDSvxIgpmoitKnjJG9ODIQ8/WjzBhsptYfc4/WF961ymr82GX4L/U0L28HuEA=="
ENDPOINT = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
CATEGORIES = ["PTY", "RN1", "T1H"]

os.makedirs(SAVE_DIR, exist_ok=True)

def collect_weather(nx, ny, base_dt):
    """지정된 base_dt 기준으로 날씨 수집, 실패 시 30분 이전 재시도 (최대 3회)"""
    for _ in range(3):
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

        try:
            response = requests.get(ENDPOINT, params=params, timeout=5)
            if response.status_code == 200 and response.text.strip():
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
        except:
            pass

        base_dt -= timedelta(minutes=30)
        time.sleep(0.3)

    return None

def main():
    # 타임스탬프: 2025-04-24 05:30 ~ 2025-04-25 00:30
    start_time = datetime(2025, 4, 24, 5, 30)
    end_time = datetime(2025, 4, 25, 0, 30)
    current = start_time

    with open(COORDS_PATH, "r", encoding="utf-8") as f:
        coords = json.load(f)

    while current <= end_time:
        timestamp = current.strftime("%Y%m%d_%H%M")
        save_path = os.path.join(SAVE_DIR, f"{timestamp}.json")
        print(f"{timestamp} 수집 시작...")

        collected = {}
        for nx_ny in coords:
            nx, ny = map(int, nx_ny.split("_"))
            result = collect_weather(nx, ny, current)
            collected[nx_ny] = result
            time.sleep(0.2)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(collected, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {save_path}\n")

        current += timedelta(minutes=30)

if __name__ == "__main__":
    main()