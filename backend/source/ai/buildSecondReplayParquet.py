# backend/source/ai/buildSecondReplayParquet.py

# === Modules ===
import os
import sys
import json
import time
import pandas
import argparse
import datetime
import multiprocessing

# === BASE ENVs SETTING ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

MEAN_NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "node")
WEATHER_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")
BUS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
POS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_pos")
TRAFFIC_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_congestion")
ROUTE_NODES_PAIR_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_pair")

LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
NX_NY_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")

SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "second_train", "replay")
os.makedirs(SAVE_PATH, exist_ok=True)

# === sub Functions Load ===
from source.utils.getDayType import getDayType
from source.utils.getTimeGroup import getTimeGroup
from source.utils.normalize import normalize
from source.utils.timeToSinCos import time_to_sin_cos
from source.utils.fallbackWeather import fallback_weather
from source.utils.extractRouteInfo import extract_route_info

def init_worker(w1, w2, w3, w4, w5, w6):
    global mean_node, stdid_number, nx_ny_stops, label_bus
    global weather_all, traffic_all

    mean_node, stdid_number, nx_ny_stops, label_bus = w1, w2, w3, w4
    weather_all, traffic_all = w5, w6

# === Process single file :: Multiprocessing ===
def process_single_file(task):
    stdid, fname, raw_date = task

    rows = []
    bus_number, direction, branch = extract_route_info(stdid, stdid_number, label_bus)
    hhmm = fname.replace(".json", "").split("_")[-1]
    trip_group_id = f"{raw_date}_{hhmm}_{stdid}"

    bus_path = os.path.join(BUS_DIR, stdid)
    pos_path = os.path.join(POS_DIR, stdid)

    with open(os.path.join(bus_path, fname), encoding='utf-8') as f:
        raw_bus_log = json.load(f)
    bus_log = raw_bus_log.get("stop_reached_logs", [])
    if len(bus_log) < 2: return []

    with open(os.path.join(pos_path, fname), encoding='utf-8') as f:
        pos_log = json.load(f)
    if len(pos_log) < 2: return []

    with open(os.path.join(ROUTE_NODES_PAIR_DIR, f"{stdid}.json"), encoding='utf-8') as f:
        route_nodes_pair = json.load(f)
    # route node reverse mapping
    route_nodes_ord = {}
    for ord_str, node_list in route_nodes_pair.items():
        ord_int = int(ord_str)
        for node_id in node_list:
            route_nodes_ord[node_id] = ord_int

    basetime = datetime.datetime.strptime(bus_log[0]['time'], "%Y-%m-%d %H:%M:%S")
    day_type = getDayType(basetime)
    weekday = {"weekday": 1, "saturday": 2, "holiday": 3}[day_type]
    timegroup = getTimeGroup(basetime)
    wd_tg = weekday * 8 + (timegroup - 1)
    max_ord = max([r['ord'] for r in bus_log])

    

    return rows

# === Main Preprocessing Function ===
def build_second_replay_parquet(target_date):
    print(f"[INFO] 시작: {target_date} replay 전처리 [Second Model]")

    mean_date = (datetime.datetime.strptime(target_date, "%Y%m%d") - 
                 datetime.timedelta(days = 2)).strftime("%Y%m%d") # mean = the day before
    raw_date = (datetime.datetime.strptime(target_date, "%Y%m%d") - 
                 datetime.timedelta(days = 1)).strftime("%Y%m%d") # raw = yesterday
    
    # Load files
    with open(os.path.join(MEAN_NODE_DIR, f"{mean_date}.json"), encoding="utf-8") as f:
        mean_node = json.load(f)
    with open(STDID_NUMBER_PATH, encoding='utf-8') as f:
        stdid_number = json.load(f)
    with open(NX_NY_STOP_PATH, encoding='utf-8') as f:
        nx_ny_stops = json.load(f)
    with open(LABEL_BUS_PATH, encoding="utf-8") as f:
        label_bus = json.load(f)
    
    weather_all = {}
    for file in os.listdir(WEATHER_DIR):
        if file.endswith(".json") and file.startswith(raw_date):
            key = file.replace(".json", "")
            with open(os.path.join(WEATHER_DIR, file), encoding='utf-8') as f:
                weather_all[key] = json.load(f)
    
    traffic_all = {}
    for file in os.listdir(TRAFFIC_DIR):
        if file.endswith(".json") and file.startswith(raw_date):
            key = file.replace(".json", "")
            with open(os.path.join(TRAFFIC_DIR, file), encoding='utf-8') as f:
                traffic_all[key] = json.load(f)

    # prepare to execute
    task_list = []
    for stdid in os.listdir(BUS_DIR):
        bus_path = os.path.join(BUS_DIR, stdid)
        for fname in os.listdir(bus_path):
            if not fname.startswith(raw_date): continue
            fname.replace(".json", "")
            task_list.append((stdid, fname, raw_date))
    
    with multiprocessing.Pool(multiprocessing.cpu_count(),
                              initializer=init_worker,
                              initargs=(mean_node, stdid_number, nx_ny_stops, label_bus,
                                        weather_all, traffic_all)
                             ) as pool:
        results = pool.map(process_single_file, task_list)

    rows = [row for group in results for row in group]
    df = pandas.DataFrame(rows)
    df.to_parquet(os.path.join(SAVE_PATH, f"{target_date}.parquet"), index=False)
    print(f"[INFO] 전처리 완료. 저장 위치: {os.path.join(SAVE_PATH, target_date + '.parquet')}")

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required = True, help = "Target date in YYYYmmDD format")
    args = parser.parse_args()

    now = time.time()
    build_second_replay_parquet(args.date)
    print("총 소요 시간: ", round(time.time() - now, 1), "sec")