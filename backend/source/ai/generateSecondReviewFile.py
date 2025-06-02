# backend/source/ai/generateSecondReviewFile.py

# === Modules ===
import os
import sys
import json
import time
import torch
import pandas as pd
import argparse
import datetime
import multiprocessing

# === BASE ENVs SETTING ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

MEAN_NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "node")
WEATHER_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")
TRAFFIC_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "traffic")
BUS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
POS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_pos")
ROUTE_NODES_PAIR_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_pair")
ROUTE_NODES_MAPPED_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_mapped")

LAST_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "last_stop.json")
LAST_NODE_PATH = os.path.join(BASE_DIR, "data", "processed", "last_node.json")
LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
NX_NY_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")

SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "second_train", "self_review")
os.makedirs(SAVE_PATH, exist_ok=True)

# === sub Functions Load ===
from inferenceSecondModel import infer_eta_batch
from inferenceSecondModel import prepare_input_tensor
from inferenceSecondModel import load_second_eta_model
from source.utils.getDayType import getDayType
from source.utils.getTimeGroup import getTimeGroup
from source.utils.normalize import normalize
from source.utils.timeToSinCos import time_to_sin_cos
from source.utils.fallbackWeather import fallback_weather
from source.utils.extractRouteInfo import extract_route_info
from source.utils.getAvgCongestion import get_avg_congestion_list

# model = load_second_eta_model("/your/path/model.pth", device)
# results = infer_eta_batch(model, [row1, row2, ...], device)

def init_worker(w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11):
    global mean_node, stdid_number, nx_ny_stops, label_bus
    global weather_all, last_stop, last_node, traffic_all
    global route_nodes_mapped, route_nodes_pair, route_nodes_ord

    mean_node, stdid_number, nx_ny_stops, label_bus = w1, w2, w3, w4
    weather_all, last_stop, last_node, traffic_all = w5, w6, w7, w8
    route_nodes_mapped, route_nodes_pair, route_nodes_ord = w9, w10, w11

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

    basetime = datetime.datetime.strptime(bus_log[0]['time'], "%Y-%m-%d %H:%M:%S")
    day_type = getDayType(basetime)
    weekday = {"weekday": 1, "saturday": 2, "holiday": 3}[day_type]
    timegroup = getTimeGroup(basetime)
    wd_tg = weekday * 8 + (timegroup - 1)
    max_ord = last_stop[stdid]
    max_node = last_node[stdid]
    dep_sin, dep_cos = time_to_sin_cos(basetime)

    # === Preprocess start for row ===
    prev_node = None
    for record in pos_log:
        # process for unique NODE_ID(route node)
        node_block = record.get("matched_route_node")
        if not node_block: continue
        now_node = node_block.get("NODE_ID") # node id ratio
        if now_node is None: continue
        if now_node == prev_node: continue
        prev_node = now_node
        node_id_ratio = round(now_node / max_node, 4)

        now_ord = route_nodes_ord[stdid][now_node]
        arr_time = datetime.datetime.strptime(record["time"], "%Y-%m-%d %H:%M:%S")
        mask = [1, 1, 1, 1, 1]

        # === Prepare to process each nodes ===
        mn_dict = mean_node.get(stdid, {}).get(str(now_node), {})
        if not mn_dict: continue # avoid zero means

        # === X Features ===
        mn_total_list = [0 for _ in range(5)]
        mn_weekday_list = [0 for _ in range(5)]
        mn_timegroup_list = [0 for _ in range(5)]
        mn_wd_tg_list = [0 for _ in range(5)]
        pty_list = [0 for _ in range(5)]
        rn1_list = [0.0 for _ in range(5)]
        t1h_list = [0.0 for _ in range(5)]
        ord_ratio_list = [0.0 for _ in range(5)]
        y_list = [0.0 for _ in range(5)]
        avg_congestion_list = get_avg_congestion_list(
            now_ord, max_ord, now_node, route_nodes_pair, route_nodes_mapped, traffic_all, arr_time, stdid
        )

        # === Process for each ORDs ===
        for i in range(1, 6):
            # check range
            target_ord = now_ord + i
            if target_ord > max_ord:
                mask[i - 1] = 0
                continue

            # if mean info not exists for ord
            ord_mn_dict = mn_dict.get(str(i), {})
            if not ord_mn_dict:
                mask[i - 1] = 0
                continue

            raw_mn_total = ord_mn_dict.get("total", {}).get("mean", None)
            raw_mn_weekday = ord_mn_dict.get(f"weekday_{weekday}", {}).get("mean", None)
            raw_mn_timegroup = ord_mn_dict.get(f"timegroup_{timegroup}", {}).get("mean", None)
            raw_mn_wd_tg = ord_mn_dict.get(f"wd_tg_{weekday}_{timegroup}", {}).get("mean", None)

            # fallback logic
            mn_total = normalize(raw_mn_total, 0, 3000) if raw_mn_total is not None else 0.0
            if raw_mn_wd_tg is not None: mn_wd_tg = normalize(raw_mn_wd_tg, 0, 3000)
            elif raw_mn_weekday is not None: mn_wd_tg = normalize(raw_mn_weekday, 0, 3000)
            elif raw_mn_timegroup is not None: mn_wd_tg = normalize(raw_mn_timegroup, 0, 3000)
            else: mn_wd_tg = mn_total
            mn_weekday = normalize(raw_mn_weekday, 0, 3000) if raw_mn_weekday is not None else mn_total
            mn_timegroup = normalize(raw_mn_timegroup, 0, 3000) if raw_mn_timegroup is not None else mn_total

            mn_total_list[i - 1] = mn_total
            mn_weekday_list[i - 1] = mn_weekday
            mn_timegroup_list[i - 1] = mn_timegroup
            mn_wd_tg_list[i - 1] = mn_wd_tg

            nx_ny = nx_ny_stops.get(f"{stdid}_{target_ord}", "63_89")
            weather = fallback_weather(arr_time, nx_ny, weather_all)
            pty_list[i - 1] = weather["PTY"]
            rn1_list[i - 1] = weather["RN1"]
            t1h_list[i - 1] = weather["T1H"]

            ord_node_id = route_nodes_pair[stdid][str(target_ord)][0] # oid {i} ratio
            ord_ratio_list[i - 1] = round(ord_node_id / max_node, 4)

            target_time_str = next((item["time"] for item in bus_log if item["ord"] == target_ord), None)
            if target_time_str is None:
                mask[i - 1] = 0
                continue
            target_time = datetime.datetime.strptime(target_time_str, "%Y-%m-%d %H:%M:%S")

            y_list[i - 1] = normalize((target_time - arr_time).total_seconds(), 0, 3000)
        
        # === Compose Features per Row ===
        row = {
            "trip_group_id": trip_group_id,
            "ord": now_ord,
            "node_id": now_node,
            "x_bus_number": bus_number,
            "x_direction": direction,
            "x_branch": branch,
            "x_weekday": weekday,
            "x_timegroup": timegroup,
            "x_weekday_timegroup": wd_tg,
            "x_departure_time_sin": dep_sin,
            "x_departure_time_cos": dep_cos,
            "x_node_id_ratio": node_id_ratio
        }

        for i in range(5):
            idx = i + 1
            row[f"y_{idx}"] = y_list[i]
            row[f"mask_{idx}"] = mask[i]
            row[f"x_mean_interval_{idx}_total"] = mn_total_list[i]
            row[f"x_mean_interval_{idx}_weekday"] = mn_weekday_list[i]
            row[f"x_mean_interval_{idx}_timegroup"] = mn_timegroup_list[i]
            row[f"x_mean_interval_{idx}_weekday_timegroup"] = mn_wd_tg_list[i]
            row[f"x_ord_{idx}_ratio"] = ord_ratio_list[i]
            row[f"x_weather_{idx}_PTY"] = pty_list[i]
            row[f"x_weather_{idx}_RN1"] = rn1_list[i]
            row[f"x_weather_{idx}_T1H"] = t1h_list[i]
            row[f"x_avg_{idx}_congestion"] = avg_congestion_list[i]
            row[f"x_prev_pred_elapsed_{idx}"] = 0.0

        rows.append(row)

    return rows

# === Main Preprocessing Function ===
def build_second_review_base(target_date):

    print(f"[INFO] 시작: {target_date} review basefile 전처리 [Second Model]")

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
    with open(LAST_STOP_PATH, encoding='utf-8') as f:
        last_stop = json.load(f)
    with open(LAST_NODE_PATH, encoding='utf-8') as f:
        last_node = json.load(f)

    traffic_all = {}
    for file in os.listdir(TRAFFIC_DIR):
        if file.endswith(".json") and file.startswith(raw_date):
            key = file.replace(".json", "")
            with open(os.path.join(TRAFFIC_DIR, file), encoding='utf-8') as f:
                records = json.load(f)
                traffic_all[key] = {}
                for item in records:
                    traffic_all[key][(item["id"], item["sub"])] = item["grade"]

    weather_all = {}
    for file in os.listdir(WEATHER_DIR):
        if file.endswith(".json") and file.startswith(raw_date):
            key = file.replace(".json", "")
            with open(os.path.join(WEATHER_DIR, file), encoding='utf-8') as f:
                weather_all[key] = json.load(f)
    
    route_nodes_mapped = {}
    for file in os.listdir(ROUTE_NODES_MAPPED_DIR):
        if file.endswith(".json"):
            stdid = file.replace(".json", "")
            with open(os.path.join(ROUTE_NODES_MAPPED_DIR, file), encoding='utf-8') as f:
                route_nodes_mapped[stdid] = json.load(f).get("resultList", []) # by stdid
    
    route_nodes_pair = {}
    route_nodes_ord = {}
    for file in os.listdir(ROUTE_NODES_PAIR_DIR):
        if file.endswith(".json"):
            stdid = file.replace(".json", "")
            with open(os.path.join(ROUTE_NODES_PAIR_DIR, file), encoding='utf-8') as f:
                route_nodes_pair[stdid] = json.load(f) # by stdid

                route_nodes_ord[stdid] = {}
                for ord_str, node_list in route_nodes_pair[stdid].items():
                    for node_id in node_list:
                        route_nodes_ord[stdid][int(node_id)] = int(ord_str)

    # prepare to execute
    task_list = []
    for stdid in os.listdir(BUS_DIR):
        bus_path = os.path.join(BUS_DIR, stdid)
        for fname in os.listdir(bus_path):
            if not fname.startswith(raw_date): continue
            if not fname.endswith(".json"): continue
            task_list.append((stdid, fname, raw_date))
    
    with multiprocessing.Pool(multiprocessing.cpu_count(),
                              initializer=init_worker,
                              initargs=(mean_node, stdid_number, nx_ny_stops, label_bus,
                                        weather_all, last_stop, last_node, traffic_all,
                                        route_nodes_mapped, route_nodes_pair, route_nodes_ord)
                             ) as pool:
        results = pool.map(process_single_file, task_list)

    rows = [row for group in results for row in group]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(BASE_DIR, "data", "model", "secondETA", "replay", f"{raw_date}.pth")
    model = load_second_eta_model(model_path, device)
    
    # === Inference :: applied mini-batch ===
    batch_size = 2048
    preds = []

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]

        x_batch = []
        for row in batch:
            x_raw = {k.replace("x_", ""): v for k, v in row.items() if k.startswith("x_")}
            x_tensor = prepare_input_tensor(x_raw, device)  # ← 여기서 tensor화
            x_batch.append(x_tensor)

        batch_preds = infer_eta_batch(model, x_batch, device)
        preds.extend(batch_preds)

    for row, pred_list in zip(rows, preds):
        for i in range(5):
            row[f"x_prev_pred_elapsed_{i+1}"] = pred_list[i]

    save_file = os.path.join(SAVE_PATH, f"{raw_date}.json")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"[INFO] Saved review base JSON: {save_file}")

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required = True, help = "Target date in YYYYmmDD format")
    args = parser.parse_args()

    now = time.time()
    build_second_review_base(args.date)
    print("총 소요 시간: ", round(time.time() - now, 1), "sec")