# backend/source/ai/buildReviewParquet.py

import os
import sys
import json
import math
import time
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# ==== 경로 설정 ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from source.utils.getDayType import getDayType

# ==== 상수 설정 ====
STOPS_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
LABEL_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")
MEAN_ELAPSED_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
MEAN_INTERVAL_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "interval")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")
ETA_TABLE_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", "first_model")
RAW_LOG_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
SAVE_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "self_review")
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== 보조 함수 ====
def normalize(v, min_val, max_val):
    return max(min((v - min_val) / (max_val - min_val), 1), 0)

def time_to_sin_cos(dt):
    minutes = dt.hour * 60 + dt.minute
    angle = 2 * math.pi * (minutes / 1440)
    return math.sin(angle), math.cos(angle)

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

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_route_info(stdid, stdid_number, label_bus):
    route_str = stdid_number[str(stdid)]
    bus_number_str = route_str[:-2]
    direction = 0 if route_str[-2] == 'A' else 1
    branch = int(route_str[-1])
    bus_number = int(label_bus.get(bus_number_str, 0))
    return bus_number, direction, branch

def fallback_forecast(target_dt, nx_ny, forecast_all):
    fallback_keys = [(target_dt - timedelta(hours=h)).strftime('%Y%m%d_%H00') for h in range(0, 6)]
    fallback_dates = [target_dt.strftime('%Y%m%d'),
                      (target_dt - timedelta(days=1)).strftime('%Y%m%d'),
                      (target_dt - timedelta(days=2)).strftime('%Y%m%d')]

    for date_key in fallback_dates:
        forecast = forecast_all.get(date_key)
        if not forecast:
            continue
        for ts in fallback_keys:
            if ts in forecast:
                nx, ny = map(int, nx_ny.split('_'))
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if abs(dx) + abs(dy) != 1:
                            continue
                        key2 = f"{nx+dx}_{ny+dy}"
                        data = forecast[ts].get(key2)
                        if data and all(k in data for k in ("PTY", "TMP", "PCP")):
                            return {
                                "PTY": int(data["PTY"]),
                                "T1H": normalize(data["TMP"], -30, 50),
                                "RN1": normalize(data["PCP"], 0, 100)
                            }
    return {"PTY": 0, "T1H": normalize(20.0, -30, 50), "RN1": normalize(0.0, 0, 100)}

def process_single_review(args):
    stdid, date_str, eta_table, stdid_number, label_bus, label_stop, mean_elapsed, mean_interval, forecast_all = args
    target_date = datetime.strptime(date_str, "%Y%m%d")
    eta_date = (target_date - timedelta(days=1)).strftime("%Y%m%d")
    wd_label = {"weekday": 1, "saturday": 2, "holiday": 3}[getDayType(target_date)]
    path = os.path.join(RAW_LOG_DIR, stdid, f"{eta_date}_0000.json")

    if not os.path.exists(path):
        return []

    try:
        trips = load_json(path)
    except:
        return []

    rows = []
    for trip in trips:
        hhmm = trip[0].split()[-1][:5]
        key = f"{stdid}_{hhmm}"
        if key not in eta_table:
            continue

        bn, dr, br = extract_route_info(stdid, stdid_number, label_bus)
        stops = trip
        eta_dict = eta_table[key]
        dep = datetime.strptime(eta_dict["1"], "%Y-%m-%d %H:%M:%S")

        for i in range(1, len(stops)):
            ord = i + 1
            stop = stops[i]
            prev_eta = datetime.strptime(eta_dict["1"], "%Y-%m-%d %H:%M:%S")
            curr_eta = datetime.strptime(eta_dict[str(ord)], "%Y-%m-%d %H:%M:%S")
            prev_elapsed = (curr_eta - prev_eta).total_seconds()

            stop_id = stop["STOP_ID"]
            tg = get_timegroup(dep)
            wd_tg = wd_label * 8 + (tg - 1)
            stop_idx = int(label_stop.get(str(stop_id), 0))
            nx_ny = f"{stop['NODE_ID']//1000000}_{(stop['NODE_ID']%1000000)//1000}"
            weather = fallback_forecast(dep, nx_ny, forecast_all)
            me = mean_elapsed.get(str(stdid), {}).get(str(ord), {})
            mi = mean_interval.get(str(stop_id), {})

            row = {
                "trip_group_id": key,
                "ord": ord,
                "x_bus_number": bn,
                "x_direction": dr,
                "x_branch": br,
                "x_weekday": wd_label,
                "x_timegroup": tg,
                "x_weekday_timegroup": wd_tg,
                "x_mean_elapsed_total": normalize(me.get("total", {}).get("mean", -1), 0, 7200),
                "x_mean_elapsed_weekday": normalize(me.get(f"weekday_{wd_label}", {}).get("mean", -1), 0, 7200),
                "x_mean_elapsed_timegroup": normalize(me.get(f"timegroup_{tg}", {}).get("mean", -1), 0, 7200),
                "x_mean_elapsed_wd_tg": normalize(me.get(f"wd_tg_{wd_label}_{tg}", {}).get("mean", -1), 0, 7200),
                "x_node_id": stop_idx,
                "x_mean_interval_total": normalize(mi.get("total", {}).get("mean", -1), 0, 600),
                "x_mean_interval_weekday": normalize(mi.get(f"weekday_{wd_label}", {}).get("mean", -1), 0, 600),
                "x_mean_interval_timegroup": normalize(mi.get(f"timegroup_{tg}", {}).get("mean", -1), 0, 600),
                "x_mean_interval_wd_tg": normalize(mi.get(f"wd_tg_{wd_label}_{tg}", {}).get("mean", -1), 0, 600),
                "x_weather_PTY": weather["PTY"],
                "x_weather_T1H": weather["T1H"],
                "x_weather_RN1": weather["RN1"],
                "x_departure_time_sin": time_to_sin_cos(dep)[0],
                "x_departure_time_cos": time_to_sin_cos(dep)[1],
                "x_ord_ratio": round(ord / len(stops), 4),
                "x_prev_pred_elapsed": normalize(prev_elapsed, 0, 7200),
                "y": normalize(stop["ELAPSED"], 0, 7200)
            }
            rows.append(row)

    return rows

def save_review_parquet(date_str):
    target_date = datetime.strptime(date_str, "%Y%m%d")
    eta_date = (target_date - timedelta(days=1)).strftime("%Y%m%d")
    mean_date = (target_date - timedelta(days=2)).strftime("%Y%m%d")

    stdid_number = load_json(STDID_NUMBER_PATH)
    label_bus = load_json(LABEL_BUS_PATH)
    label_stop = load_json(LABEL_STOP_PATH)
    mean_elapsed = load_json(os.path.join(MEAN_ELAPSED_DIR, f"{mean_date}.json"))
    mean_interval = load_json(os.path.join(MEAN_INTERVAL_DIR, f"{mean_date}.json"))
    forecast_all = {f.replace(".json", ""): load_json(os.path.join(FORECAST_DIR, f)) for f in os.listdir(FORECAST_DIR)}
    eta_table = load_json(os.path.join(ETA_TABLE_DIR, f"{eta_date}.json"))

    task_args = [
        (stdid, date_str, eta_table, stdid_number, label_bus, label_stop, mean_elapsed, mean_interval, forecast_all)
        for stdid in os.listdir(RAW_LOG_DIR)
    ]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_review, task_args)

    rows = [row for group in results for row in group]
    df = pd.DataFrame(rows)
    save_path = os.path.join(SAVE_DIR, f"{date_str}.parquet")
    df.to_parquet(save_path, index=False)
    print(f"[INFO] Saved self-review parquet: {save_path} | rows: {len(df)}")

# ==== main ====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    now = time.time()
    print("Self-review parquet 생성 중... ", args.date)
    save_review_parquet(args.date)
    print("소요 시간: ", time.time() - now, "sec")