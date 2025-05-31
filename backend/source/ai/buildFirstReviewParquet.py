# backend/source/ai/buildFirstReviewParquet.py

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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")
MEAN_ELAPSED_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
MEAN_INTERVAL_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "interval")

LAST_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "last_stop.json")
LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
LABEL_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")
NX_NY_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
STOP_TO_ROUTES_PATH = os.path.join(BASE_DIR, "data", "processed", "stop_to_routes.json")
ETA_TABLE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", "first_model")

SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "self_review")
os.makedirs(SAVE_PATH, exist_ok=True)

# === 보조 함수 ===
from source.utils.normalize import normalize
from source.utils.getDayType import getDayType
from source.utils.fallbackForecast import fallback_forecast
from source.utils.getTimeGroup import getTimeGroup
from source.utils.normalize import normalize
from source.utils.timeToSinCos import time_to_sin_cos

def init_worker(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10):
    global forecast_all, ord_lookup, stdid_number, nx_ny_stops, mean_elapsed
    global mean_interval, label_bus, label_stops, eta_table, last_stop
    
    forecast_all, ord_lookup, stdid_number, nx_ny_stops, mean_elapsed = w1, w2, w3, w4, w5
    mean_interval, label_bus, label_stops, eta_table, last_stop = w6, w7, w8, w9, w10

# === 개별 파일 처리 함수 ===
def process_single_file(args):
    stdid, fname, target_date = args

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
    tg = getTimeGroup(base_time)
    wd_tg = weekday * 8 + (tg - 1)
    trip_group_id = f"{target_date}_{hhmm}_{stdid}"
    eta_key = f"{stdid}_{hhmm.replace(':', '')}"
    eta_dict = eta_table.get(eta_key, {})
    max_ord = last_stop[stdid]

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

        # mean
        me = mean_elapsed.get(str(stdid), {}).get(str(ord), {})
        mi = mean_interval.get(str(stop_id), {})
        pme = mean_elapsed.get(str(stdid), {}).get(str(ord - 1), {})
        if not me or not mi or not pme: continue

        raw_me_total = me.get("total", {}).get("mean", None)
        raw_me_wd_tg = me.get(f"wd_tg_{weekday}_{tg}", {}).get("mean", None)
        raw_me_weekday = me.get(f"weekday_{weekday}", {}).get("mean", None)
        raw_me_timegroup = me.get(f"timegroup_{tg}", {}).get("mean", None)

        me_total = normalize(raw_me_total, 0, 7200) if raw_me_total is not None else 0.0
        if raw_me_wd_tg is not None: me_wd_tg = normalize(raw_me_wd_tg, 0, 7200)
        elif raw_me_weekday is not None: me_wd_tg = normalize(raw_me_weekday, 0, 7200)
        elif raw_me_timegroup is not None: me_wd_tg = normalize(raw_me_timegroup, 0, 7200)
        else: me_wd_tg = me_total

        me_weekday = normalize(raw_me_weekday, 0, 7200) if raw_me_weekday is not None else me_total
        me_timegroup = normalize(raw_me_timegroup, 0, 7200) if raw_me_timegroup is not None else me_total

        raw_mi_total = mi.get("total", {}).get("mean", None)
        raw_mi_wd_tg = mi.get(f"wd_tg_{weekday}_{tg}", {}).get("mean", None)
        raw_mi_weekday = mi.get(f"weekday_{weekday}", {}).get("mean", None)
        raw_mi_timegroup = mi.get(f"timegroup_{tg}", {}).get("mean", None)
        
        mi_total = normalize(raw_mi_total, 0, 600) if raw_mi_total is not None else 0.0
        if raw_mi_wd_tg is not None: mi_wd_tg = normalize(raw_mi_wd_tg, 0, 600)
        elif raw_mi_weekday is not None: mi_wd_tg = normalize(raw_mi_weekday, 0, 600)
        elif raw_mi_timegroup is not None: mi_wd_tg = normalize(raw_mi_timegroup, 0, 600)
        else: mi_wd_tg = mi_total

        mi_weekday = normalize(raw_mi_weekday, 0, 600) if raw_mi_weekday is not None else mi_total        
        mi_timegroup = normalize(raw_mi_timegroup, 0, 600) if raw_mi_timegroup is not None else mi_total

        raw_pme_total = pme.get("total", {}).get("mean", None)
        raw_pme_wd_tg = pme.get(f"wd_tg_{weekday}_{tg}", {}).get("mean", None)
        raw_pme_weekday = pme.get(f"weekday_{weekday}", {}).get("mean", None)
        raw_pme_timegroup = pme.get(f"timegroup_{tg}", {}).get("mean", None)

        pme_total = normalize(raw_pme_total, 0, 7200) if raw_pme_total is not None else 0.0
        if raw_pme_wd_tg is not None: pme_wd_tg = normalize(raw_pme_wd_tg, 0, 7200)
        elif raw_pme_weekday is not None: pme_wd_tg = normalize(raw_pme_weekday, 0, 7200)
        elif raw_pme_timegroup is not None: pme_wd_tg = normalize(raw_pme_timegroup, 0, 7200)
        else: pme_wd_tg = pme_total
        
        pme_weekday = normalize(raw_pme_weekday, 0, 7200) if raw_pme_weekday is not None else pme_total       
        pme_timegroup = normalize(raw_pme_timegroup, 0, 7200) if raw_pme_timegroup is not None else pme_total

        nx_ny = nx_ny_stops.get(f"{stdid}_{ord}", "63_89")
        weather = fallback_forecast(arr_time, nx_ny, forecast_all)

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
            "x_prev_mean_elapsed_total": pme_total,
            "x_prev_mean_elapsed_weekday": pme_weekday,
            "x_prev_mean_elapsed_timegroup": pme_timegroup,
            "x_prev_mean_elapsed_wd_tg": pme_wd_tg,
            "x_node_id": stop_idx,
            "x_mean_interval_total": mi_total,
            "x_mean_interval_weekday": mi_weekday,
            "x_mean_interval_timegroup": mi_timegroup,
            "x_mean_interval_wd_tg": mi_wd_tg,
            "x_weather_PTY": weather['PTY'],
            "x_weather_RN1": normalize(weather['RN1'], 0, 100),
            "x_weather_T1H": normalize(weather['T1H'], -30, 50),
            "x_departure_time_sin": dep_sin,
            "x_departure_time_cos": dep_cos,
            "x_ord_ratio": round(ord / max_ord, 4),
            "x_prev_ord_ratio": round((ord-1) / max_ord, 4),
            "x_prev_pred_elapsed": normalize(prev_pred_elapsed, 0, 7200)
        }
        rows.append(row)
    return rows

def build_review_parquet(target_date):
    print(f"[INFO] Self-review parquet 생성 중...  {target_date}")
    mean_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=2)).strftime("%Y%m%d")
    raw_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")

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
    with open(os.path.join(ETA_TABLE_PATH, f"{raw_date}.json"), encoding='utf-8') as f:
        eta_table = json.load(f)
    with open(LAST_STOP_PATH, encoding='utf-8') as f:
        last_stop = json.load(f)

    forecast_all = {}
    for file in os.listdir(FORECAST_DIR):
        if file.endswith(".json") and (file.startswith(target_date) or file.startswith(raw_date) or file.startswith(mean_date)):
            key = file.replace(".json", "")
            with open(os.path.join(FORECAST_DIR, file), encoding='utf-8') as f:
                forecast_all[key] = json.load(f)

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
            task_list.append((stdid, fname, target_date))

    with Pool(cpu_count(),
              initializer=init_worker,
              initargs=(forecast_all, ord_lookup, stdid_number, nx_ny_stops, mean_elapsed, 
                        mean_interval, label_bus, label_stops, eta_table, last_stop)
              ) as pool:
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
    print("총 소요 시간:", round(time.time() - now, 1), "sec")