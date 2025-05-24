# backend/source/ai/buildReplayParquet.py

import os
import sys
import json
import time
import math
import argparse
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# === 환경 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
LABEL_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")
sys.path.append(BASE_DIR)

from source.utils.getDayType import getDayType

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
MEAN_ELAPSED_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
MEAN_INTERVAL_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "interval")
STOP_TO_ROUTES_PATH = os.path.join(BASE_DIR, "data", "processed", "stop_to_routes.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
NX_NY_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
WEATHER_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")

SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "replay")
os.makedirs(SAVE_PATH, exist_ok=True)

# === 보조 함수 정의 ===
def get_time_group(departure_time):
    minutes = departure_time.hour * 60 + departure_time.minute
    if 330 <= minutes < 420: return 1
    elif 420 <= minutes < 540: return 2
    elif 540 <= minutes < 690: return 3
    elif 690 <= minutes < 840: return 4
    elif 840 <= minutes < 1020: return 5
    elif 1020 <= minutes < 1140: return 6
    elif 1140 <= minutes < 1260: return 7
    else: return 8

def normalize(value, min_val, max_val):
    return max(min((value - min_val) / (max_val - min_val), 1), 0)

def time_to_sin_cos(dt):
    minutes = dt.hour * 60 + dt.minute
    angle = 2 * math.pi * (minutes / 1440)
    return math.sin(angle), math.cos(angle)

# === 날씨 fallback용 보조 함수 ===
def fallback_weather(target_time: datetime, nx_ny: str, weather_all: dict):
    tried_timestamps = sorted(weather_all.keys(), reverse=True)

    for i in range(4):  # 최대 5 fallback까지 시도
        closest_key = None
        for ts in tried_timestamps:
            ts_dt = datetime.strptime(ts, "%Y%m%d_%H%M")
            if ts_dt <= target_time:
                closest_key = ts
                break

        if closest_key is None:
            target_time -= timedelta(minutes=10)
            continue

        grid_data = weather_all[closest_key]
        # nx_ny가 없거나 값이 None이면 fallback
        if nx_ny in grid_data and grid_data[nx_ny] is not None:
            val = grid_data[nx_ny]
            if ( val['PTY'] is not None and val['PTY'] >= 0 and
                val['RN1'] is not None and val['RN1'] >= 0 and
                val['T1H'] is not None ):
                return val

        # 인접 격자 탐색
        nx, ny = map(int, nx_ny.split("_"))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                key2 = f"{nx+dx}_{ny+dy}"
                if key2 in grid_data and grid_data[key2] is not None:
                    val2 = grid_data[key2]
                    if ( val2['PTY'] is not None and val2['PTY'] >= 0 and
                        val2['RN1'] is not None and val2['RN1'] >= 0 and
                        val2['T1H'] is not None ):
                        return val2

        target_time -= timedelta(minutes=10)

    return {"PTY": 0, "RN1": 0.0, "T1H": 20.0}  # fallback 실패 시 기본값

# === 개별 파일 처리 함수 ===
def process_single_file(args):
    stdid, fname, target_date, ord_lookup, stdid_number, nx_ny_stops, mean_elapsed, mean_interval, weather_all, label_bus, label_stops = args

    rows = []
    stdid_path = os.path.join(RAW_DIR, stdid)
    route_id = stdid_number.get(stdid, "000X0")
    bus_number_str = route_id[:-2]
    bus_number = int(label_bus.get(bus_number_str, 0))  # fallback index=0
    direction = 0 if route_id[-2] == 'A' else 1
    branch = int(route_id[-1])
    hhmm = fname.replace(".json", "").split("_")[-1]
    trip_group_id = f"{target_date}_{hhmm}_{stdid}"

    with open(os.path.join(stdid_path, fname), encoding='utf-8') as f:
        log = json.load(f)
    logs = log.get("stop_reached_logs", [])
    if len(logs) < 2: return []

    base_time = datetime.strptime(logs[0]['time'], "%Y-%m-%d %H:%M:%S")
    day_type = getDayType(base_time)
    weekday = {"weekday": 1, "saturday": 2, "holiday": 3}[day_type]
    timegroup = get_time_group(base_time)
    wd_tg = weekday * 8 + (timegroup - 1)
    max_ord = max([r['ord'] for r in logs])

    for record in logs[1:]:
        ord = record['ord']
        arr_time = datetime.strptime(record['time'], "%Y-%m-%d %H:%M:%S")
        real_elapsed = (arr_time - base_time).total_seconds()

        me_dict = mean_elapsed.get(stdid, {}).get(str(ord), {})
        pme_dict = mean_elapsed.get(stdid, {}).get(str(ord - 1), {})

        raw_me_total = me_dict.get("total", {}).get("mean", None)
        me_total = normalize(raw_me_total, 0, 7200) if raw_me_total is not None else 0.0
        raw_pme_total = pme_dict.get("total", {}).get("mean", None)
        pme_total = normalize(raw_pme_total, 0, 7200) if raw_pme_total is not None else 0.0

        raw_me_weekday = me_dict.get(f"weekday_{weekday}", {}).get("mean", None)
        me_weekday = normalize(raw_me_weekday, 0, 7200) if raw_me_weekday is not None else me_total
        raw_pme_weekday = pme_dict.get(f"weekday_{weekday}", {}).get("mean", None)
        pme_weekday = normalize(raw_pme_weekday, 0, 7200) if raw_pme_weekday is not None else pme_total

        raw_me_timegroup = me_dict.get(f"timegroup_{timegroup}", {}).get("mean", None)
        me_timegroup = normalize(raw_me_timegroup, 0, 7200) if raw_me_timegroup is not None else me_total
        raw_pme_timegroup = pme_dict.get(f"timegroup_{timegroup}", {}).get("mean", None)
        pme_timegroup = normalize(raw_pme_timegroup, 0, 7200) if raw_pme_timegroup is not None else pme_total

        raw_me_wd_tg = me_dict.get(f"wd_tg_{weekday}_{timegroup}", {}).get("mean", None)
        if raw_me_wd_tg is not None:
            me_wd_tg = normalize(raw_me_wd_tg, 0, 7200)
        elif raw_me_weekday is not None:
            me_wd_tg = me_weekday
        elif raw_me_timegroup is not None:
            me_wd_tg = me_timegroup
        else:
            me_wd_tg = me_total

        raw_pme_wd_tg = pme_dict.get(f"wd_tg_{weekday}_{timegroup}", {}).get("mean", None)
        if raw_pme_wd_tg is not None:
            pme_wd_tg = normalize(raw_pme_wd_tg, 0, 7200)
        elif raw_pme_weekday is not None:
            pme_wd_tg = pme_weekday
        elif raw_pme_timegroup is not None:
            pme_wd_tg = pme_timegroup
        else:
            pme_wd_tg = pme_total

        stop_id = ord_lookup.get((stdid, ord), None)
        if stop_id is None:
            continue
        try:
            stop_id_str = str(stop_id)  # string이어도 int로 강제 변환
            stop_id_index = int(label_stops.get(stop_id_str, 0))  # fallback index=0
        except ValueError:
            continue  # 혹시라도 이상한 값 들어올 경우 방어

        mi_dict = mean_interval.get(stop_id, {})

        raw_mi_total = mi_dict.get("total", {}).get("mean", None)
        mi_total = normalize(raw_mi_total, 0, 600) if raw_mi_total is not None else 0.0

        raw_mi_weekday = mi_dict.get(f"weekday_{weekday}", {}).get("mean", None)
        mi_weekday = normalize(raw_mi_weekday, 0, 600) if raw_mi_weekday is not None else mi_total

        raw_mi_timegroup = mi_dict.get(f"timegroup_{timegroup}", {}).get("mean", None)
        mi_timegroup = normalize(raw_mi_timegroup, 0, 600) if raw_mi_timegroup is not None else mi_total

        raw_mi_wd_tg = mi_dict.get(f"wd_tg_{weekday}_{timegroup}", {}).get("mean", None)
        if raw_mi_wd_tg is not None:
            mi_wd_tg = normalize(raw_mi_wd_tg, 0, 600)
        elif raw_mi_weekday is not None:
            mi_wd_tg = mi_weekday
        elif raw_mi_timegroup is not None:
            mi_wd_tg = mi_timegroup
        else:
            mi_wd_tg = mi_total

        nx_ny = nx_ny_stops.get(f"{stdid}_{ord}", "63_89")
        weather = fallback_weather(arr_time, nx_ny, weather_all)
        t_sin, t_cos = time_to_sin_cos(base_time)

        row = {
            "trip_group_id": trip_group_id,
            "ord": ord,
            "y": normalize(real_elapsed, 0, 7200),
            "x_bus_number": bus_number,
            "x_direction": direction,
            "x_branch": branch,
            "x_weekday": weekday,
            "x_timegroup": timegroup,
            "x_weekday_timegroup": wd_tg,
            "x_mean_elapsed_total": me_total,
            "x_mean_elapsed_weekday": me_weekday,
            "x_mean_elapsed_timegroup": me_timegroup,
            "x_mean_elapsed_wd_tg": me_wd_tg,
            "x_prev_mean_elapsed_total": pme_total,
            "x_prev_mean_elapsed_weekday": pme_weekday,
            "x_prev_mean_elapsed_timegroup": pme_timegroup,
            "x_prev_mean_elapsed_wd_tg": pme_wd_tg,
            "x_node_id": stop_id_index,
            "x_mean_interval_total": mi_total,
            "x_mean_interval_weekday": mi_weekday,
            "x_mean_interval_timegroup": mi_timegroup,
            "x_mean_interval_wd_tg": mi_wd_tg,
            "x_weather_PTY": weather['PTY'],
            "x_weather_RN1": normalize(weather['RN1'], 0, 100),
            "x_weather_T1H": normalize(weather['T1H'], -30, 50),
            "x_departure_time_sin": t_sin,
            "x_departure_time_cos": t_cos,
            "x_ord_ratio": round(ord / max_ord, 4),
            "x_prev_ord_ratio": round((ord-1) / max_ord, 4),
            "x_prev_pred_elapsed": 0.0
        }
        rows.append(row)
    return rows

# === 메인 전처리 함수 ===
def build_replay_parquet(target_date):
    print(f"[INFO] 시작: {target_date} replay 전처리")
    mean_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=2)).strftime("%Y%m%d") # mean은 이틀 전

    with open(os.path.join(MEAN_ELAPSED_DIR, f"{mean_date}.json"), encoding='utf-8') as f:
        mean_elapsed = json.load(f)
    with open(os.path.join(MEAN_INTERVAL_DIR, f"{mean_date}.json"), encoding='utf-8') as f:
        mean_interval = json.load(f)
    with open(STOP_TO_ROUTES_PATH, encoding='utf-8') as f:
        stop_to_routes = json.load(f)
    with open(STDID_NUMBER_PATH, encoding='utf-8') as f:
        stdid_number = json.load(f)
    with open(NX_NY_STOP_PATH, encoding='utf-8') as f:
        nx_ny_stops = json.load(f)
    with open(LABEL_BUS_PATH, encoding="utf-8") as f:
        label_bus = json.load(f)
    with open(LABEL_STOP_PATH, encoding="utf-8") as f:
        label_stops = json.load(f)

    # 역매핑 딕셔너리 생성 (성능 개선용)
    ord_lookup = {}  # (stdid, ord) → stop_id
    for stop_id, routes in stop_to_routes.items():
        for route in routes:
            key = (route['stdid'], route['ord'])
            ord_lookup[key] = stop_id

    weather_all = {}
    for file in os.listdir(WEATHER_DIR):
        if file.endswith(".json"):
            key = file.replace(".json", "")
            with open(os.path.join(WEATHER_DIR, file), encoding='utf-8') as f:
                weather_all[key] = json.load(f)

    # 처리 대상 파일 준비
    task_list = []
    for stdid in os.listdir(RAW_DIR):
        stdid_path = os.path.join(RAW_DIR, stdid)
        if not os.path.isdir(stdid_path): continue
        for fname in os.listdir(stdid_path):
            if not fname.endswith(".json"): continue
            raw_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
            if not fname.startswith(raw_date): continue
            task_list.append((stdid, fname, raw_date, ord_lookup, stdid_number, nx_ny_stops, mean_elapsed, mean_interval, weather_all, label_bus, label_stops))

    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_file, task_list)

    rows = [row for group in results for row in group]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(SAVE_PATH, f"{target_date}_2.parquet"), index=False)
    print(f"[INFO] 전처리 완료. 저장 위치: {os.path.join(SAVE_PATH, target_date + '_2.parquet')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Target date in YYYYMMDD format")
    args = parser.parse_args()

    now = time.time()
    build_replay_parquet(args.date)
    print("총 소요 시간: ", time.time() - now, "sec")
