# backend/source/ai/generateFirstETA.py

import os
import sys
import json
import math
import time
import torch
import argparse
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

# ==== 경로 설정 ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.getDayType import getDayType
from source.ai.FirstETAModel import FirstETAModel

# ==== 경로 상수 ====
STOPS_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
LABEL_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")
MEAN_ELAPSED_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
MEAN_INTERVAL_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "interval")
DEPARTURE_CACHE_DIR = os.path.join(BASE_DIR, "data", "processed", "departure_cache")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")

ETA_SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocess", "eta_table", "first_model")
FEATURE_PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocess", "first_train", "inference")
os.makedirs(ETA_SAVE_PATH, exist_ok=True)
os.makedirs(FEATURE_PARQUET_PATH, exist_ok=True)

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
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def extract_route_info(stdid, stdid_number, label_bus):
    route_str = stdid_number[str(stdid)]
    bus_number_str = route_str[:-2]
    direction = 0 if route_str[-2] == 'A' else 1
    branch = int(route_str[-1])
    bus_number = int(label_bus.get(bus_number_str, 0))
    return bus_number, direction, branch

def forecast_lookup(target_dt, nx_ny, forecast_all):
    target_keys = [(target_dt - timedelta(hours=h)).strftime('%Y%m%d_%H00') for h in range(0, 6)]
    fallback_dates = [target_dt.strftime('%Y%m%d'), (target_dt - timedelta(days=1)).strftime('%Y%m%d'), (target_dt - timedelta(days=2)).strftime('%Y%m%d')]
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

# ==== 단일 stdid 처리 함수 ====
def process_single_entry(args):
    entry, date_str, target_date, weekday_label, model_path, stdid_number, label_bus, label_stops, mean_elapsed, mean_interval, forecast_all = args
    hhmm = entry['time']
    dep_hour, dep_minute = map(int, hhmm.split(":"))
    dep_time = datetime(target_date.year, target_date.month, target_date.day, dep_hour, dep_minute)
    model = FirstETAModel().eval()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    rows, eta_output = [], {}
    for stdid in entry['stdid']:
        stop_path = os.path.join(STOPS_DIR, f"{stdid}.json")
        if not os.path.exists(stop_path):
            continue
        stops = load_json(stop_path)['resultList']
        bus_number, direction, branch = extract_route_info(stdid, stdid_number, label_bus)
        max_ord = max([stop['STOP_ORD'] for stop in stops])
        eta_list = []
        prev_elapsed = 0.0
        for stop in stops:
            ord = stop['STOP_ORD']
            stop_id = stop['STOP_ID']
            stop_id_str = str(stop_id)
            stop_id_index = int(label_stops.get(stop_id_str, 0))
            wd = weekday_label
            tg = get_timegroup(dep_time)
            wd_tg = wd * 8 + (tg - 1)
            nx_ny = f"{stop['NODE_ID'] // 1000000}_{(stop['NODE_ID'] % 1000000) // 1000}"
            weather = forecast_lookup(dep_time, nx_ny, forecast_all)
            me_dict = mean_elapsed.get(str(stdid), {}).get(str(ord), {})
            mi_dict = mean_interval.get(str(stop_id), {})
            row = {
                "x_bus_number": bus_number,
                "x_direction": direction,
                "x_branch": branch,
                "x_weekday": wd,
                "x_timegroup": tg,
                "x_weekday_timegroup": wd_tg,
                "x_mean_elapsed_total": normalize(me_dict.get("total", {}).get("mean", -1), 0, 7200),
                "x_mean_elapsed_weekday": normalize(me_dict.get(f"weekday_{wd}", {}).get("mean", -1), 0, 7200),
                "x_mean_elapsed_timegroup": normalize(me_dict.get(f"timegroup_{tg}", {}).get("mean", -1), 0, 7200),
                "x_mean_elapsed_wd_tg": normalize(me_dict.get(f"wd_tg_{wd}_{tg}", {}).get("mean", -1), 0, 7200),
                "x_node_id": stop_id_index,
                "x_mean_interval_total": normalize(mi_dict.get("total", {}).get("mean", -1), 0, 600),
                "x_mean_interval_weekday": normalize(mi_dict.get(f"weekday_{wd}", {}).get("mean", -1), 0, 600),
                "x_mean_interval_timegroup": normalize(mi_dict.get(f"timegroup_{tg}", {}).get("mean", -1), 0, 600),
                "x_mean_interval_wd_tg": normalize(mi_dict.get(f"wd_tg_{wd}_{tg}", {}).get("mean", -1), 0, 600),
                "x_weather_PTY": weather['PTY'],
                "x_weather_RN1": weather['RN1'],
                "x_weather_T1H": weather['T1H'],
                "x_departure_time_sin": time_to_sin_cos(dep_time)[0],
                "x_departure_time_cos": time_to_sin_cos(dep_time)[1],
                "x_ord_ratio": round(ord / max_ord, 4),
                "x_prev_pred_elapsed": normalize(prev_elapsed, 0, 7200)
            }
            
            x_tensor = {}
            for k, v in row.items():
                key = k.replace("x_", "")
                if isinstance(v, int):
                    tensor = torch.tensor([v], dtype=torch.long)
                else:
                    tensor = torch.tensor([v], dtype=torch.float32)

                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(1)  # [1] -> [1, 1]
                x_tensor[key] = tensor

            with torch.no_grad():
                pred_mean, _ = model(x_tensor)
                elapsed = float(pred_mean.item())
                prev_elapsed = elapsed
            parquet_row = row.copy()
            parquet_row['trip_group_id'] = f"{date_str}_{hhmm}_{stdid}"
            parquet_row['ord'] = ord
            rows.append(parquet_row)
            eta_time = dep_time + timedelta(seconds=elapsed)
            eta_list.append({"ord": ord, "time": eta_time.strftime("%Y-%m-%d %H:%M:%S")})
        eta_output[f"{stdid}_{hhmm}"] = eta_list
    return rows, eta_output

# === main 함수 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    now = time.time()

    target_date = datetime.strptime(args.date, "%Y%m%d")
    date_str = target_date.strftime("%Y%m%d")
    prev_date_str = (target_date - timedelta(days=1)).strftime("%Y%m%d")
    weekday_type = getDayType(target_date)
    weekday_label = {"weekday": 1, "saturday": 2, "holiday": 3}[weekday_type]

    stdid_number = load_json(STDID_NUMBER_PATH)
    label_bus = load_json(LABEL_BUS_PATH)
    label_stops = load_json(LABEL_STOP_PATH)
    mean_elapsed = load_json(os.path.join(MEAN_ELAPSED_DIR, f"{prev_date_str}.json"))
    mean_interval = load_json(os.path.join(MEAN_INTERVAL_DIR, f"{prev_date_str}.json"))
    forecast_all = {fname.replace(".json", ""): load_json(os.path.join(FORECAST_DIR, fname)) for fname in os.listdir(FORECAST_DIR) if fname.endswith(".json")}
    departure_cache = load_json(os.path.join(DEPARTURE_CACHE_DIR, f"{weekday_type}.json"))['data']

    model_path = os.path.join(BASE_DIR, "data", "model", "firstETA", "replay", f"{date_str}.pth")

    task_args = [(entry, date_str, target_date, weekday_label, model_path, stdid_number, label_bus, label_stops, mean_elapsed, mean_interval, forecast_all) for entry in departure_cache]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_single_entry, task_args)

    all_rows, all_eta = [], {}
    for rows, eta_dict in results:
        all_rows.extend(rows)
        all_eta.update(eta_dict)

    with open(os.path.join(ETA_SAVE_PATH, f"{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump(all_eta, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(all_rows)
    next_date_str = (target_date + timedelta(days=1)).strftime("%Y%m%d")
    df.to_parquet(os.path.join(FEATURE_PARQUET_PATH, f"{next_date_str}.parquet"), index=False)
    print(f"[INFO] ETA 추론 및 저장 완료: {date_str}")
    print("소요 시간: ", time.time() - now, "sec")