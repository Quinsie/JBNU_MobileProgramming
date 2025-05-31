# backend/source/ai/buildFirstReplayParquetStart.py

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

WEATHER_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
MEAN_ELAPSED_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
MEAN_INTERVAL_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "interval")

LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
NX_NY_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
LABEL_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
STOP_TO_ROUTES_PATH = os.path.join(BASE_DIR, "data", "processed", "stop_to_routes.json")

SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "replay")
os.makedirs(SAVE_PATH, exist_ok=True)

# === 보조 함수 정의 ===
from source.utils.getDayType import getDayType
from source.utils.getTimeGroup import getTimeGroup
from source.utils.normalize import normalize
from source.utils.timeToSinCos import time_to_sin_cos
from source.utils.fallbackWeather import fallback_weather

def init_worker(w1, w2, w3, w4, w5, w6, w7, w8):
    global weather_all, ord_lookup, stdid_number, nx_ny_stops, mean_elapsed
    global mean_interval, label_bus, label_stops
    
    weather_all, ord_lookup, stdid_number, nx_ny_stops, mean_elapsed = w1, w2, w3, w4, w5
    mean_interval, label_bus, label_stops = w6, w7, w8

# === 개별 파일 처리 함수 ===
def process_single_file(args):
    stdid, fname, target_date = args

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
    timegroup = getTimeGroup(base_time)
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
def build_replay_parquet(target_date, mean_date):
    print(f"[INFO] 시작: {target_date} replay 전처리")

    start_date = datetime.strptime("20250505", "%Y%m%d")
    end_date = datetime.strptime(target_date, "%Y%m%d") - timedelta(days=1)

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
    for dt in pd.date_range(start=start_date, end=end_date):
        raw_date = dt.strftime("%Y%m%d")
        for stdid in os.listdir(RAW_DIR):
                stdid_path = os.path.join(RAW_DIR, stdid)
                for fname in os.listdir(stdid_path):
                    if not fname.endswith(".json"): continue
                    if not fname.startswith(raw_date): continue
                    task_list.append((stdid, fname, raw_date))

    with Pool(cpu_count(),
              initializer=init_worker,
              initargs=(weather_all, ord_lookup, stdid_number, nx_ny_stops, mean_elapsed, 
                        mean_interval, label_bus, label_stops)
              ) as pool:
        results = pool.map(process_single_file, task_list)

    rows = [row for group in results for row in group]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(SAVE_PATH, f"{target_date}.parquet"), index=False)
    print(f"[INFO] 전처리 완료. 저장 위치: {os.path.join(SAVE_PATH, target_date + '.parquet')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Target date in YYYYMMDD format")
    args = parser.parse_args()

    now = time.time()
    target_date = args.date
    mean_date = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=2)).strftime("%Y%m%d")
    build_replay_parquet(target_date, mean_date)
    print("총 소요 시간: ", time.time() - now, "sec")