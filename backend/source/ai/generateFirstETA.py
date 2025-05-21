# backend/source/ai/generateFirstETA.py

import os
import sys
import json
import math
import time
import torch
import argparse
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
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")
DEPARTURE_CACHE_DIR = os.path.join(BASE_DIR, "data", "processed", "departure_cache")
SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", "first_model")
os.makedirs(SAVE_PATH, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = None

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

def set_global_model(model):
    global MODEL
    MODEL = model
    MODEL.eval()

# ==== 단일 stdid 처리 함수 ====
def infer_single(entry, target_date, wd_label, stdid_number, label_bus, label_stops, mean_elapsed, mean_interval, forecast_all):
    hhmm = entry['time']  # ex: "07:50"
    dep_hour, dep_minute = map(int, hhmm.split(":"))
    dep = datetime(target_date.year, target_date.month, target_date.day, dep_hour, dep_minute)
    result = {}

    global MODEL

    for stdid in entry["stdid"]:
        stop_file = os.path.join(STOPS_DIR, f"{stdid}.json")
        if not os.path.exists(stop_file): continue
        stops = load_json(stop_file)['resultList']
        bn, dr, br = extract_route_info(stdid, stdid_number, label_bus)
        max_ord = max(s["STOP_ORD"] for s in stops)
        eta_dict, prev_elapsed = {}, 0.0

        for s in stops:
            ord = s["STOP_ORD"]; stop_id = s["STOP_ID"]
            eta_time = dep + timedelta(seconds=prev_elapsed)
            if ord == 1:
                eta_dict[str(ord)] = eta_time.strftime("%Y-%m-%d %H:%M:%S")
                continue

            stop_idx = int(label_stops.get(str(stop_id), 0))
            tg = get_timegroup(dep)
            wd_tg = wd_label * 8 + (tg - 1)
            nx_ny = f"{s['NODE_ID']//1000000}_{(s['NODE_ID']%1000000)//1000}"
            weather = forecast_lookup(dep, nx_ny, forecast_all)
            me = mean_elapsed.get(str(stdid), {}).get(str(ord), {})
            mi = mean_interval.get(str(stop_id), {})

            row = {
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
                "x_weather_RN1": weather["RN1"],
                "x_weather_T1H": weather["T1H"],
                "x_departure_time_sin": time_to_sin_cos(dep)[0],
                "x_departure_time_cos": time_to_sin_cos(dep)[1],
                "x_ord_ratio": round(ord / max_ord, 4),
                "x_prev_pred_elapsed": normalize(prev_elapsed, 0, 7200)
            }

            int_keys = {
                "bus_number", "direction", "branch", "node_id", "weekday", "timegroup", "weekday_timegroup", "weather_PTY"
            }
            float_keys = {
                "mean_elapsed_total", "mean_elapsed_weekday", "mean_elapsed_timegroup", "mean_elapsed_wd_tg",
                "mean_interval_total", "mean_interval_weekday", "mean_interval_timegroup", "mean_interval_wd_tg",
                "weather_RN1", "weather_T1H",
                "departure_time_sin", "departure_time_cos",
                "ord_ratio", "prev_pred_elapsed"
            }

            x_tensor = {}
            for k, v in row.items():
                key = k.replace("x_", "")

                if key in float_keys:
                    val = torch.tensor([v], dtype=torch.float32)  # shape: (1, 1)
                elif key in int_keys:
                    val = torch.tensor([v], dtype=torch.long)     # shape: (1, 1)
                
                if val.dim() == 1 and val.dtype == torch.float32:
                    val = val.unsqueeze(1)

                x_tensor[key] = val.to(device)

            with torch.no_grad():
                pred_mean, _ = MODEL(x_tensor)
                if ord % 10 == 0:
                    print(f"[{stdid}_{hhmm}] ord {ord}: pred_mean = {pred_mean.item():.6f}")
                elapsed = float(pred_mean.item()) * 7200
                eta_time = dep + timedelta(seconds=elapsed)
                eta_dict[str(ord)] = eta_time.strftime("%Y-%m-%d %H:%M:%S")

        result[f"{stdid}_{hhmm}"] = eta_dict
    return result

# ===== main =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    now = time.time()
    print(f"ETA Table 생성 시작... {args.date}")

    target_date = datetime.strptime(args.date, "%Y%m%d")
    date_str = args.date
    prev_date_str = (target_date - timedelta(days=1)).strftime("%Y%m%d")
    weekday_type = getDayType(target_date)
    wd_label = {"weekday": 1, "saturday": 2, "holiday": 3}[weekday_type]

    stdid_number = load_json(STDID_NUMBER_PATH)
    label_bus = load_json(LABEL_BUS_PATH)
    label_stops = load_json(LABEL_STOP_PATH)
    mean_elapsed = load_json(os.path.join(MEAN_ELAPSED_DIR, f"{prev_date_str}.json"))
    mean_interval = load_json(os.path.join(MEAN_INTERVAL_DIR, f"{prev_date_str}.json"))
    forecast_all = {f.replace(".json", ""): load_json(os.path.join(FORECAST_DIR, f)) for f in os.listdir(FORECAST_DIR)}
    dep_data = load_json(os.path.join(DEPARTURE_CACHE_DIR, f"{weekday_type}.json"))["data"]

    model_pth = os.path.join(BASE_DIR, "data", "model", "firstETA", "replay", f"{date_str}.pth")
    model_obj = FirstETAModel(); model_obj.load_state_dict(torch.load(model_pth, map_location=device))
    set_global_model(model_obj.to(device))

    # global model 로드 (multiprocessing-safe)
    # set_global_model(os.path.join(BASE_DIR, "data", "model", "firstETA", "replay", f"{date_str}_full.pt"))

    task_args = [(entry, target_date, wd_label, stdid_number, label_bus, label_stops, mean_elapsed, mean_interval, forecast_all) for entry in dep_data]
    
    def unpack_and_infer(args): return infer_single(*args)
    # with Pool(cpu_count()) as pool:
    #     results = pool.map(unpack_and_infer, task_args)
    results = [unpack_and_infer(args) for args in task_args]

    final = {}; [final.update(r) for r in results]
    with open(os.path.join(SAVE_PATH, f"{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"[INFO] ETA 저장 완료: {date_str} / 총 {len(final)}개 운행")
    print("소요 시간: ", time.time() - now, "sec")