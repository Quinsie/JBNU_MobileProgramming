# tmp.py

import os
import sys
import json
import math
import time
import torch
import random
from datetime import datetime, timedelta

# ==== 경로 설정 ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.getDayType import getDayType
from source.ai.FirstETAModel import FirstETAModel

# ==== 기본 설정 ====
DATE = "20250507"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 50

# ==== 경로 상수 ====
STOPS_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
LABEL_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")
MEAN_ELAPSED_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
MEAN_INTERVAL_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "interval")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")
DEPARTURE_CACHE_DIR = os.path.join(BASE_DIR, "data", "processed", "departure_cache")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model", "firstETA", "replay", f"{DATE}.pth")

# ==== 유틸 함수들 ====
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

def forecast_lookup(dep, nx_ny, forecast_all, wd_label):
    target_keys = [(dep - timedelta(hours=h)).strftime('%Y%m%d_%H00') for h in range(0, 4)]
    fallback_dates = [dep.strftime('%Y%m%d'), (dep - timedelta(days=1)).strftime('%Y%m%d'), (dep - timedelta(days=2)).strftime('%Y%m%d')]
    for date_key in fallback_dates:
        forecast = forecast_all.get(date_key)
        if not forecast: continue
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

# ==== 추론 시작 ====
print("[INFO] Running ETA sample inference test")
target_date = datetime.strptime(DATE, "%Y%m%d")
prev_date_str = (target_date - timedelta(days=1)).strftime("%Y%m%d")
weekday_type = getDayType(target_date)
wd_label = {"weekday": 1, "saturday": 2, "holiday": 3}[weekday_type]

# Load all data
stdid_number = load_json(STDID_NUMBER_PATH)
label_bus = load_json(LABEL_BUS_PATH)
label_stops = load_json(LABEL_STOP_PATH)
mean_elapsed = load_json(os.path.join(MEAN_ELAPSED_DIR, f"{prev_date_str}.json"))
mean_interval = load_json(os.path.join(MEAN_INTERVAL_DIR, f"{prev_date_str}.json"))
forecast_all = {f.replace(".json", ""): load_json(os.path.join(FORECAST_DIR, f)) for f in os.listdir(FORECAST_DIR)}
dep_data = load_json(os.path.join(DEPARTURE_CACHE_DIR, f"{weekday_type}.json"))["data"]

# Load model
model = FirstETAModel(); model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# Sample entries
sample_entries = random.sample(dep_data, min(SAMPLE_SIZE, len(dep_data)))

for entry in sample_entries:
    print(f"\n--- ETA Inference for {entry['stdid']} @ {entry['time']} ---")
    hhmm = entry['time']
    dep_hour, dep_minute = map(int, hhmm.split(":"))
    dep = datetime(target_date.year, target_date.month, target_date.day, dep_hour, dep_minute)

    for stdid in entry['stdid']:
        stop_file = os.path.join(STOPS_DIR, f"{stdid}.json")
        if not os.path.exists(stop_file): continue
        stops = load_json(stop_file)['resultList']
        bn, dr, br = extract_route_info(stdid, stdid_number, label_bus)
        max_ord = max(s["STOP_ORD"] for s in stops)
        prev_elapsed = 0.0

        for s in stops:
            ord = s["STOP_ORD"]; stop_id = s["STOP_ID"]
            if ord == 1: continue
            tg = get_timegroup(dep)
            wd_tg = wd_label * 8 + (tg - 1)
            nx_ny = f"{s['NODE_ID']//1000000}_{(s['NODE_ID']%1000000)//1000}"
            weather = forecast_lookup(dep, nx_ny, forecast_all, wd_label)

            me = mean_elapsed.get(str(stdid), {}).get(str(ord), {})
            mi = mean_interval.get(str(stop_id), {})

            def get_mean(m, key, default):
                raw = m.get(key, {}).get("mean", None)
                return normalize(raw, *default) if raw is not None else 0.0

            row = {
                "x_bus_number": bn,
                "x_direction": dr,
                "x_branch": br,
                "x_weekday": wd_label,
                "x_timegroup": tg,
                "x_weekday_timegroup": wd_tg,
                "x_mean_elapsed_total": get_mean(me, "total", (0, 7200)),
                "x_mean_elapsed_weekday": get_mean(me, f"weekday_{wd_label}", (0, 7200)),
                "x_mean_elapsed_timegroup": get_mean(me, f"timegroup_{tg}", (0, 7200)),
                "x_mean_elapsed_wd_tg": get_mean(me, f"wd_tg_{wd_label}_{tg}", (0, 7200)),
                "x_node_id": int(label_stops.get(str(stop_id), 0)),
                "x_mean_interval_total": get_mean(mi, "total", (0, 600)),
                "x_mean_interval_weekday": get_mean(mi, f"weekday_{wd_label}", (0, 600)),
                "x_mean_interval_timegroup": get_mean(mi, f"timegroup_{tg}", (0, 600)),
                "x_mean_interval_wd_tg": get_mean(mi, f"wd_tg_{wd_label}_{tg}", (0, 600)),
                "x_weather_PTY": weather["PTY"],
                "x_weather_RN1": weather["RN1"],
                "x_weather_T1H": weather["T1H"],
                "x_departure_time_sin": time_to_sin_cos(dep)[0],
                "x_departure_time_cos": time_to_sin_cos(dep)[1],
                "x_ord_ratio": round(ord / max_ord, 4),
                "x_prev_pred_elapsed": 0.0
            }

            x_tensor = {}
            for k, v in row.items():
                key = k.replace("x_", "")
                dtype = torch.float32 if isinstance(v, float) else torch.long
                val = torch.tensor([v], dtype=dtype)
                if dtype == torch.float32: val = val.unsqueeze(1)
                x_tensor[key] = val.to(DEVICE)

            with torch.no_grad():
                pred_mean, _ = model(x_tensor)
                elapsed = float(pred_mean.item()) * 7200
                print(f"  ORD {ord:02d}: ETA+{elapsed:.1f}s")