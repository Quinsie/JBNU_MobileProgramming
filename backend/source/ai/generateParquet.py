# backend/source/ai/generateParquet.py
# 1차 ETA 학습용 parquet 자동 생성 스크립트

import os
import sys
import json
import pandas as pd
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.convertToGrid import convert_to_grid
from source.utils.getDayType import getDayType
from source.utils.logger import log

# 경로 설정
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
ETA_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table")
NXNY_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
WEATHER_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")
SAVE_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
os.makedirs(SAVE_DIR, exist_ok=True)

# 날짜 계산 (오늘 기준: 25일 새벽 실행 → 24일 raw, 23일 ETA 사용)
today = datetime.now().date()
yesterday = today - timedelta(days=1)
two_days_ago = today - timedelta(days=2)
start_time = time.time()

YMD = yesterday.strftime("%Y%m%d")
ETA_YMD = two_days_ago.strftime("%Y%m%d")

# ETA 테이블 불러오기
eta_path = os.path.join(ETA_DIR, f"{ETA_YMD}.json")
with open(eta_path, encoding="utf-8") as f:
    eta_table = json.load(f)

# nx_ny 매핑 불러오기
with open(NXNY_PATH, encoding="utf-8") as f:
    nxny_map = json.load(f)

# 날씨 타임스탬프 매핑
weather_ts_map = {
    datetime.strptime(f.replace(".json", ""), "%Y%m%d_%H%M"): f
    for f in os.listdir(WEATHER_DIR) if f.endswith(".json")
}
weather_cache = {}
for ts, fname in weather_ts_map.items():
    try:
        with open(os.path.join(WEATHER_DIR, fname), encoding="utf-8") as wf:
            weather_cache[ts] = json.load(wf)
    except:
        continue

def process_file(args):
    stdid, file = args
    rows = []
    departure = file[9:13]  # HHMM
    key = f"{stdid}_{departure}"
    eta_dict = eta_table.get(key)
    if not eta_dict:
        return rows

    path = os.path.join(RAW_DIR, stdid, file)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stop_logs = data.get("stop_reached_logs", [])
    day_type = getDayType(datetime.strptime(YMD, "%Y%m%d"))
    start_hour, start_minute = int(departure[:2]), int(departure[2:])
    departure_minute = start_hour * 60 + start_minute

    for stop in stop_logs:
        ord = str(stop.get("ord"))
        actual_time = stop.get("time")
        if ord not in eta_dict:
            continue

        try:
            eta_dt = datetime.strptime(f"{YMD} {eta_dict[ord]}", "%Y%m%d %H:%M:%S")
            actual_dt = datetime.strptime(actual_time, "%Y-%m-%d %H:%M:%S")
            elapsed = int((actual_dt - eta_dt).total_seconds())
        except:
            continue

        nx_ny = nxny_map.get(f"{stdid}_{ord}")
        PTY = RN1 = T1H = None
        if nx_ny:
            weather_key = max([ts for ts in weather_cache if ts <= actual_dt], default=None)
            if weather_key:
                weather = weather_cache.get(weather_key, {}).get(nx_ny)
                if weather:
                    PTY = int(weather["PTY"])
                    RN1 = float(weather["RN1"])
                    T1H = float(weather["T1H"])

        rows.append({
            "route_id": int(stdid),
            "departure_time": departure_minute,
            "day_type": {"weekday": 0, "saturday": 1, "holiday": 2}[day_type],
            "stop_order": int(ord),
            "PTY": PTY,
            "RN1": RN1,
            "T1H": T1H,
            "target_elapsed_time": elapsed
        })

    return rows

def generate_parquet():
    log("generateParquet", f"{YMD} 전처리 시작")
    all_args = []
    for stdid in os.listdir(RAW_DIR):
        route_path = os.path.join(RAW_DIR, stdid)
        for file in os.listdir(route_path):
            if not file.startswith(YMD):
                continue
            all_args.append((stdid, file))

    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, all_args)

    flat_rows = [row for group in results for row in group]
    if not flat_rows:
        log("generateParquet", f"데이터 없음: {YMD}")
        return

    df = pd.DataFrame(flat_rows)
    save_path = os.path.join(SAVE_DIR, f"eta_train_{YMD}.parquet")
    df.to_parquet(save_path, index=False)
    log("generateParquet", f"{len(df)} rows 저장 완료 → {save_path}")
    log("generateParquet", f"전체 소요시간: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    generate_parquet()