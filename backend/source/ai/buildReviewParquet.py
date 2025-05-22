# backend/source/ai/buildReviewParquet.py

import os
import sys
import json
import math
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.getDayType import getDayType

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
MEAN_ELAPSED_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
MEAN_INTERVAL_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "interval")
STOP_TO_ROUTES_PATH = os.path.join(BASE_DIR, "data", "processed", "stop_to_routes.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
NX_NY_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
LABEL_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")
ETA_TABLE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", "first_model")
SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "self_review")
os.makedirs(SAVE_PATH, exist_ok=True)

# === 보조 함수 ===
def normalize(value, min_val, max_val):
    return max(min((value - min_val) / (max_val - min_val), 1), 0)

def get_timegroup(dt):
    minutes = dt.hour * 60 + dt.minute
    if 330 <= minutes < 420: return 1
    elif 420 <= minutes < 540: return 2
    elif 540 <= minutes < 690: return 3
    elif 690 <= minutes < 840: return 4
    elif 840 <= minutes < 1020: return 5
    elif 1020 <= minutes < 1140: return 6
    elif 1140 <= minutes < 1260: return 7
    else: return 8

def time_to_sin_cos(dt):
    minutes = dt.hour * 60 + dt.minute
    angle = 2 * math.pi * (minutes / 1440)
    return math.sin(angle), math.cos(angle)

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def forecast_lookup(target_dt, nx_ny, forecast_all):
    target_keys = [(target_dt - timedelta(hours=h)).strftime('%Y%m%d_%H00') for h in range(0, 4)]
    fallback_dates = [
        target_dt.strftime('%Y%m%d'),
        (target_dt - timedelta(days=1)).strftime('%Y%m%d'),
        (target_dt - timedelta(days=2)).strftime('%Y%m%d')
    ]
    for date_key in fallback_dates:
        forecast = forecast_all.get(date_key)
        if not forecast:
            continue
        for ts in target_keys:
            if ts in forecast:
                nx, ny = map(int, nx_ny.split('_'))
                for dx in [0, -1, 1]:
                    for dy in [0, -1, 1]:
                        key2 = f"{nx + dx}_{ny + dy}"
                        data = forecast[ts].get(key2)
                        if data and all(k in data for k in ('TMP', 'PCP', 'PTY')):
                            return {
                                'T1H': normalize(data['TMP'], -30, 50),
                                'RN1': normalize(data['PCP'], 0, 100),
                                'PTY': int(data['PTY'])
                            }
    return {'T1H': normalize(20.0, -30, 50), 'RN1': normalize(0.0, 0, 100), 'PTY': 0}

def process_single_file(args):
    stdid, fname, target_date, ord_lookup, eta_table, label_bus, label_stops, stdid_number, nx_ny_stops, mean_elapsed, mean_interval, forecast_all = args

    rows = []
    route_info = stdid_number[str(stdid)]
    bus_number_str, direction, branch = route_info[:-2], 0 if route_info[-2] == 'A' else 1, int(route_info[-1])
    bus_number = int(label_bus.get(bus_number_str, 0))
    raw_path = os.path.join(RAW_DIR, stdid, fname)
    with open(raw_path, encoding='utf-8') as f:
        log = json.load(f)

    logs = log.get("stop_reached_logs", [])
    if len(logs) < 2: return []

    base_time = datetime.strptime(logs[0]['time'], "%Y-%m-%d %H:%M:%S")
    hhmm = base_time.strftime("%H:%M")
    dep_sin, dep_cos = time_to_sin_cos(base_time)
    weekday_type = getDayType(base_time)
    weekday = {"weekday": 1, "saturday": 2, "holiday": 3}[weekday_type]
    tg = get_timegroup(base_time)
    wd_tg = weekday * 8 + (tg - 1)
    trip_group_id = f"{target_date}_{hhmm}_{stdid}"
    eta_key = f"{stdid}_{hhmm.replace(':', '')}"
    eta_dict = eta_table.get(eta_key, {})
    max_ord = max([r['ord'] for r in logs])

    for record in logs[1:]:
        ord = record['ord']
        arr_time = datetime.strptime(record['time'], "%Y-%m-%d %H:%M:%S")
        real_elapsed = (arr_time - base_time).total_seconds()
        prev_pred = eta_dict.get(str(ord))
        if prev_pred is None: continue
        prev_pred_elapsed = (datetime.strptime(prev_pred, "%Y-%m-%d %H:%M:%S") - base_time).total_seconds()

        stop_id = ord_lookup.get((stdid, ord), None)
        if stop_id is None: continue
        stop_idx = int(label_stops.get(str(stop_id), 0))
        nx_ny = nx_ny_stops.get(f"{stdid}_{ord}", "63_89")
        weather = forecast_lookup(arr_time, nx_ny, forecast_all)

        me = mean_elapsed.get(str(stdid), {}).get(str(ord), {})
        mi = mean_interval.get(str(stop_id), {})

        me_total = normalize(me.get("total", {}).get("mean", -1), 0, 7200)
        me_weekday = normalize(me.get(f"weekday_{weekday}", {}).get("mean", me_total), 0, 7200)
        me_timegroup = normalize(me.get(f"timegroup_{tg}", {}).get("mean", me_total), 0, 7200)
        me_wd_tg_raw = me.get(f"wd_tg_{weekday}_{tg}", {}).get("mean", None)
        if me_wd_tg_raw is not None:
            me_wd_tg = normalize(me_wd_tg_raw, 0, 7200)
        elif me.get(f"weekday_{weekday}"):
            me_wd_tg = me_weekday
        elif me.get(f"timegroup_{tg}"):
            me_wd_tg = me_timegroup
        else:
            me_wd_tg = me_total

        mi_total = normalize(mi.get("total", {}).get("mean", -1), 0, 600)
        mi_weekday = normalize(mi.get(f"weekday_{weekday}", {}).get("mean", mi_total), 0, 600)
        mi_timegroup = normalize(mi.get(f"timegroup_{tg}", {}).get("mean", mi_total), 0, 600)
        mi_wd_tg_raw = mi.get(f"wd_tg_{weekday}_{tg}", {}).get("mean", None)
        if mi_wd_tg_raw is not None:
            mi_wd_tg = normalize(mi_wd_tg_raw, 0, 600)
        elif mi.get(f"weekday_{weekday}"):
            mi_wd_tg = mi_weekday
        elif mi.get(f"timegroup_{tg}"):
            mi_wd_tg = mi_timegroup
        else:
            mi_wd_tg = mi_total

        row = {
            "trip_group_id": trip_group_id,
            "ord": ord,
            "y": normalize(real_elapsed, 0, 7200),
            "x_bus_number": bus_number,
            "x_direction": direction,
            "x_branch": branch,
            "x_weekday": weekday,
            "x_timegroup": tg,
            "x_weekday_timegroup": wd_tg,
            "x_mean_elapsed_total": me_total,
            "x_mean_elapsed_weekday": me_weekday,
            "x_mean_elapsed_timegroup": me_timegroup,
            "x_mean_elapsed_wd_tg": me_wd_tg,
            "x_node_id": stop_idx,
            "x_mean_interval_total": mi_total,
            "x_mean_interval_weekday": mi_weekday,
            "x_mean_interval_timegroup": mi_timegroup,
            "x_mean_interval_wd_tg": mi_wd_tg,
            "x_weather_PTY": weather['PTY'],
            "x_weather_RN1": weather['RN1'],
            "x_weather_T1H": weather['T1H'],
            "x_departure_time_sin": dep_sin,
            "x_departure_time_cos": dep_cos,
            "x_ord_ratio": round(ord / max_ord, 4),
            "x_prev_pred_elapsed": normalize(prev_pred_elapsed, 0, 7200)
        }
        rows.append(row)
    return rows

def build_review_parquet(target_date):
    print(f"[INFO] Self-review parquet 생성 중...  {target_date}")
    mean_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=2)).strftime("%Y%m%d")
    raw_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")

    mean_elapsed = load_json(os.path.join(MEAN_ELAPSED_DIR, f"{mean_date}.json"))
    mean_interval = load_json(os.path.join(MEAN_INTERVAL_DIR, f"{mean_date}.json"))
    stop_to_routes = load_json(STOP_TO_ROUTES_PATH)
    stdid_number = load_json(STDID_NUMBER_PATH)
    nx_ny_stops = load_json(NX_NY_STOP_PATH)
    label_bus = load_json(LABEL_BUS_PATH)
    label_stops = load_json(LABEL_STOP_PATH)
    forecast_all = {f.replace(".json", ""): load_json(os.path.join(FORECAST_DIR, f)) for f in os.listdir(FORECAST_DIR)}
    eta_table = load_json(os.path.join(ETA_TABLE_PATH, f"{raw_date}.json"))

    ord_lookup = {}
    for stop_id, routes in stop_to_routes.items():
        for r in routes:
            ord_lookup[(r['stdid'], r['ord'])] = stop_id

    task_list = []
    for stdid in os.listdir(RAW_DIR):
        stdid_path = os.path.join(RAW_DIR, stdid)
        if not os.path.isdir(stdid_path): continue
        for fname in os.listdir(stdid_path):
            if not fname.endswith(".json"): continue
            if not fname.startswith(raw_date): continue
            task_list.append((stdid, fname, target_date, ord_lookup, eta_table, label_bus, label_stops, stdid_number, nx_ny_stops, mean_elapsed, mean_interval, forecast_all))

    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_file, task_list)

    rows = [r for group in results for r in group]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(SAVE_PATH, f"{target_date}.parquet"), index=False)
    print(f"[INFO] 저장 완료: {target_date}.parquet / 총 {len(df)}개 rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Target date in YYYYMMDD format")
    args = parser.parse_args()
    now = time.time()
    build_review_parquet(args.date)
    print("총 소요 시간:", time.time() - now, "sec")