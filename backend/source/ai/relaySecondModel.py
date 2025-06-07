# backend/source/ai/relaySecondModel.py

import os
import sys
import time
import json
import torch
from queue import Empty
from datetime import date, datetime, timedelta
from multiprocessing import Queue
from multiprocessing.managers import BaseManager

# === BASE ENVs SETTING ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

MEAN_NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "node")
WEATHER_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")
TRAFFIC_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "traffic")
ROUTE_NODES_PAIR_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_pair")
ROUTE_NODES_MAPPED_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_mapped")

LAST_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "last_stop.json")
LAST_NODE_PATH = os.path.join(BASE_DIR, "data", "processed", "last_node.json")
LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
NX_NY_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")

from source.utils.logger import log
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

# === Queue Server Manager Settings ===
q = Queue()

class InboundQueueManager(BaseManager): pass
InboundQueueManager.register("get_queue", callable=lambda: q)

def start_queue_server():
    manager = InboundQueueManager(address=("localhost", 50000), authkey=b"abc")
    server = manager.get_server()
    print("[RELAY] Inbound Queue 서버 시작됨 (50000)")
    server.serve_forever()

# === Sub Function ===
class OutboundQueueManager(BaseManager): pass
OutboundQueueManager.register("get_queue")

def send_to_queue(payload: dict, log_prefix: str):
    try:
        manager = OutboundQueueManager(address=("localhost", 49000), authkey=b"def")
        manager.connect()
        que = manager.get_queue()
        que.put(payload)
        log("relaySecondModel", f"{log_prefix} 전송 완료: {payload}")
    except Exception as e:
        log("relaySecondModel", f"{log_prefix} 전송 실패: {payload.get('stdid', '?')}_{e}")

def update_weather_dict(weather_all: dict, raw_date: str):
    for file in os.listdir(WEATHER_DIR):
        if file.endswith(".json") and file.startswith(raw_date):
            key = file.replace(".json", "")
            if key not in weather_all:
                path = os.path.join(WEATHER_DIR, file)
                with open(path, encoding="utf-8") as f:
                    weather_all[key] = json.load(f)

def update_traffic_dict(traffic_all: dict, raw_date: str):
    for file in os.listdir(TRAFFIC_DIR):
        if file.endswith(".json") and file.startswith(raw_date):
            key = file.replace(".json", "")
            if key not in traffic_all:
                path = os.path.join(TRAFFIC_DIR, file)
                with open(path, encoding="utf-8") as f:
                    records = json.load(f)
                    traffic_all[key] = {(item["id"], item["sub"]): item["grade"] for item in records}

# === Process Main Function ===
def consume_loop():
    print("[RELAY] Queue 소비 시작")
    today = date.today() # SET DATE TODAY
    raw_date = datetime.strftime(today, "%Y%m%d")
    mean_date = (today - timedelta(days = 1)).strftime("%Y%m%d")
    last_weather_update_time = 0
    last_traffic_update_time = 0

    # = SET INFERENCE MODEL =
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(BASE_DIR, "data", "model", "secondETA", "replay", f"{raw_date}.pth")
    model = load_second_eta_model(model_path, device)

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

    while True:
        now = time.time()
        if now - last_weather_update_time > 60:
            last_weather_update_time = now
            update_weather_dict(weather_all, raw_date)
        if now - last_traffic_update_time > 60:
            last_traffic_update_time = now
            update_traffic_dict(traffic_all, raw_date)

        try:
            row = q.get(timeout = 1)
            type = row.get("type", -1)
            if type == 0: # route node
                stdid = row.get("stdid", ""); dep_time = row.get("dep_time", "")
                timestamp = row.get("timestamp", ""); node_id = row.get("node_id", "")

                #stdid, fname, rawdate
                # === Preprocessing ===
                bus_number, direction, branch = extract_route_info(stdid, stdid_number, label_bus)
                hhmm = dep_time.replace(":", "")
                trip_group_id = f"{raw_date}_{hhmm}_{stdid}"
                basetime = datetime.strptime(f"{today} {dep_time}", "%Y-%m-%d %H:%M")
                day_type = getDayType(today)
                weekday = {"weekday": 1, "saturday": 2, "holiday": 3}[day_type]  
                timegroup = getTimeGroup(basetime)
                wd_tg = weekday * 8 + (timegroup - 1)
                max_ord = last_stop[stdid]
                max_node = last_node[stdid]
                dep_sin, dep_cos = time_to_sin_cos(basetime)

                now_node = node_id
                node_id_ratio = round(now_node / max_node, 4)
                now_ord = route_nodes_ord[stdid][now_node]
                arr_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
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
                
                xRow = {
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
                    xRow[f"mask_{idx}"] = mask[i]
                    xRow[f"x_mean_interval_{idx}_total"] = mn_total_list[i]
                    xRow[f"x_mean_interval_{idx}_weekday"] = mn_weekday_list[i]
                    xRow[f"x_mean_interval_{idx}_timegroup"] = mn_timegroup_list[i]
                    xRow[f"x_mean_interval_{idx}_weekday_timegroup"] = mn_wd_tg_list[i]
                    xRow[f"x_ord_{idx}_ratio"] = ord_ratio_list[i]
                    xRow[f"x_weather_{idx}_PTY"] = pty_list[i]
                    xRow[f"x_weather_{idx}_RN1"] = rn1_list[i]
                    xRow[f"x_weather_{idx}_T1H"] = t1h_list[i]
                    xRow[f"x_avg_{idx}_congestion"] = avg_congestion_list[i]
                    xRow[f"x_prev_pred_elapsed_{idx}"] = 0.0
                
                x_raw = {k.replace("x_", ""): v for k, v in xRow.items() if k.startswith("x_")}
                x_tensor = prepare_input_tensor(x_raw, device)
                pred = infer_eta_batch(model, [x_tensor], device)

                send_to_queue({
                    "type": 0,
                    "stdid": stdid,
                    "dep_time": dep_time,
                    "timestamp": timestamp,
                    "node_id": node_id,
                    "ord": now_ord,
                    "pred": pred
                }, log_prefix=f"TYPE {type}")

            elif type in [1, 2, 3]:
                send_to_queue(row, log_prefix=f"TYPE {type}")

            else: # error
                log("relaySecondModel", "Wrong type recieved")
        except Empty:
            time.sleep(0.5)

if __name__ == "__main__":
    from threading import Thread
    Thread(target=start_queue_server, daemon=True).start()
    consume_loop()