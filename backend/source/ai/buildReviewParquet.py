# backend/source/ai/buildReviewParquet.py

import os
import sys
import json
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# === 환경 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.getDayType import getDayType

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
ETA_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")
SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "self_review")
os.makedirs(SAVE_PATH, exist_ok=True)

# === 보조 함수 정의 ===
def normalize(value, min_val, max_val):
    return max(min((value - min_val) / (max_val - min_val), 1), 0)

def fallback_forecast(target_time: datetime, nx_ny: str, forecast_all: dict):
    fallback_hours = [0, 1, 2, 3, 4, 5]
    for days_back in range(3):
        date = (target_time - timedelta(days=days_back)).strftime("%Y%m%d")
        if date not in forecast_all:
            continue
        forecasts = forecast_all[date]

        timestamps = sorted(forecasts.keys())
        for fallback in fallback_hours:
            ts_try = target_time - timedelta(hours=fallback)
            ts_str = ts_try.strftime("%Y%m%d_%H%M")
            if ts_str not in forecasts:
                continue

            grid_data = forecasts[ts_str]
            if nx_ny in grid_data and grid_data[nx_ny] is not None:
                val = grid_data[nx_ny]
                if all(val.get(k) is not None for k in ['TMP', 'PTY', 'PCP']):
                    return val

            nx, ny = map(int, nx_ny.split("_"))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    key2 = f"{nx+dx}_{ny+dy}"
                    if key2 in grid_data and grid_data[key2] is not None:
                        val2 = grid_data[key2]
                        if all(val2.get(k) is not None for k in ['TMP', 'PTY', 'PCP']):
                            return val2
    return {"PTY": 0, "PCP": 0.0, "TMP": 20.0}  # 기본값

# === 병합 처리 함수 ===
def merge_eta_raw(args):
    stdid, hhmm, eta_ords, target_date, forecast_all = args
    raw_path = os.path.join(RAW_DIR, stdid, f"{target_date}_{hhmm}.json")
    if not os.path.exists(raw_path): return []

    with open(raw_path, encoding='utf-8') as f:
        raw = json.load(f)
    raw_logs = {str(log['ord']): log['time'] for log in raw.get("stop_reached_logs", [])}
    if "1" not in raw_logs: return []

    base_time = datetime.strptime(raw_logs["1"], "%Y-%m-%d %H:%M:%S")

    rows = []
    for ord_str, eta_time_str in eta_ords.items():
        if ord_str not in raw_logs: continue

        eta_time = datetime.strptime(f"{target_date} {eta_time_str}", "%Y-%m-%d %H:%M:%S")
        prev_pred_elapsed = (eta_time - base_time).total_seconds()
        arr_time = datetime.strptime(raw_logs[ord_str], "%Y-%m-%d %H:%M:%S")
        real_elapsed = (arr_time - base_time).total_seconds()

        nx_ny = "63_89"  # 기본 위치 (추후 nx_ny_stops 연결 시 수정 가능)
        weather = fallback_forecast(arr_time, nx_ny, forecast_all)

        row = {
            "trip_group_id": f"{target_date}_{hhmm}_{stdid}",
            "ord": int(ord_str),
            "y": normalize(real_elapsed, 0, 7200),
            "x_prev_pred_elapsed": normalize(prev_pred_elapsed, 0, 7200),
            "x_weather_PTY": weather['PTY'],
            "x_weather_RN1": normalize(weather['PCP'], 0, 100),
            "x_weather_T1H": normalize(weather['TMP'], -30, 50)
        }
        rows.append(row)
    return rows

# === 메인 전처리 함수 ===
def build_review_parquet(target_date):
    print(f"[INFO] 시작: {target_date} self-review 전처리")

    previous_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")

    forecast_all = {}
    for fname in os.listdir(FORECAST_DIR):
        if not fname.endswith(".json"): continue
        date = fname.replace(".json", "")
        with open(os.path.join(FORECAST_DIR, fname), encoding='utf-8') as f:
            forecast_all[date] = json.load(f)
    print(f"[INFO] 총 {len(forecast_all)}개 예보 파일 로드 완료")

    eta_path = os.path.join(ETA_PATH, f"{previous_date}.json")
    if not os.path.exists(eta_path):
        print(f"[ERROR] ETA Table {eta_path} 없음")
        return
    with open(eta_path, encoding='utf-8') as f:
        eta_table = json.load(f)
    print(f"[INFO] ETA Table 로드 완료: {len(eta_table)}개 그룹")

    task_list = []
    for key, ord_dict in eta_table.items():
        stdid, hhmm = key.split("_")[-2:]  # 뒤에서 2개 추출
        task_list.append((stdid, hhmm, ord_dict, previous_date, forecast_all))

    with Pool(cpu_count()) as pool:
        results = pool.map(merge_eta_raw, task_list)

    rows = [r for group in results for r in group]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(SAVE_PATH, f"{target_date}.parquet"), index=False)
    print(f"[INFO] 저장 완료: {os.path.join(SAVE_PATH, f'{target_date}.parquet')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Target date in YYYYMMDD format")
    args = parser.parse_args()

    now = time.time()
    build_review_parquet(args.date)
    print("총 소요 시간: ", time.time() - now, "sec")