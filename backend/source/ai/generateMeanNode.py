# backend/source/ai/generateMeanNode.py

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict
from multiprocessing import Pool

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
POS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_pos")
BUS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
PAIR_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_pair")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "node")
os.makedirs(SAVE_DIR, exist_ok=True)

from source.utils.getDayType import getDayType

# === 시간대 및 요일 그룹핑 ===
def get_time_group(dt):
    m = dt.hour * 60 + dt.minute
    if 330 <= m < 420: return 1
    elif 420 <= m < 540: return 2
    elif 540 <= m < 690: return 3
    elif 690 <= m < 840: return 4
    elif 840 <= m < 1020: return 5
    elif 1020 <= m < 1140: return 6
    elif 1140 <= m < 1260: return 7
    else: return 8

def get_weekday_type(dt):
    return {"weekday": 1, "saturday": 2, "holiday": 3}[getDayType(dt)]

# === 파일 하나 처리 ===
def process_file(args):
    stdid, fname, route_pair = args
    pos_path = os.path.join(POS_DIR, stdid, fname)
    bus_path = os.path.join(BUS_DIR, stdid, fname)

    try:
        with open(pos_path, "r", encoding="utf-8") as f:
            pos_data = json.load(f)
        with open(bus_path, "r", encoding="utf-8") as f:
            bus_data = json.load(f)
    except:
        return []

    pos_first_time = {}  # node_id -> 첫 감지 시간
    for row in pos_data:
        try:
            nid = row["matched_route_node"]["NODE_ID"]
            if nid not in pos_first_time:
                pos_first_time[nid] = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
        except:
            continue

    bus_time_by_ord = {}  # ord -> 도착 시간
    for row in bus_data.get("stop_reached_logs", []):
        try:
            ord_val = int(row["ord"])
            bus_time_by_ord[ord_val] = datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S")
        except:
            continue

    departure_dt = list(bus_time_by_ord.values())[0] if bus_time_by_ord else None
    if not departure_dt:
        return []
    wd = str(get_weekday_type(departure_dt))
    tg = str(get_time_group(departure_dt))
    group = f"wd_tg_{wd}_{tg}"

    result = []
    for ord_str, node_list in route_pair.items():
        try:
            ord_base = int(ord_str)
        except:
            continue

        for node_id in node_list:
            for offset in range(1, 6):  # ORD+1 ~ ORD+5
                ord_target = ord_base + offset
                if ord_target not in bus_time_by_ord:
                    continue
                if node_id not in pos_first_time:
                    continue
                elapsed = (bus_time_by_ord[ord_target] - pos_first_time[node_id]).total_seconds()
                result.append(((stdid, node_id, ord_target, group, wd, tg), elapsed))
    return result

# === 메인 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--mode", required=True, choices=["init", "append"])
    args = parser.parse_args()

    TARGET_DATE = args.date
    MODE = args.mode
    SAVE_PATH = os.path.join(SAVE_DIR, f"{TARGET_DATE}.json")

    tasks = []
    for stdid in os.listdir(PAIR_DIR):
        pair_path = os.path.join(PAIR_DIR, stdid)
        if not os.path.isfile(pair_path):
            continue
        with open(pair_path, "r", encoding="utf-8") as f:
            route_pair = json.load(f)

        pos_subdir = os.path.join(POS_DIR, stdid)
        bus_subdir = os.path.join(BUS_DIR, stdid)
        if not os.path.isdir(pos_subdir) or not os.path.isdir(bus_subdir):
            continue

        for fname in os.listdir(pos_subdir):
            if not fname.startswith(TARGET_DATE):
                continue
            if not os.path.exists(os.path.join(bus_subdir, fname)):
                continue
            tasks.append((stdid, fname, route_pair, TARGET_DATE))

    # DEBUG
    print(f"[DEBUG] 총 task 개수: {len(tasks)}")
    if tasks:
        print("[DEBUG] 예시 task:", tasks[0])

    all_results = []
    with Pool() as pool:
        for result in pool.imap_unordered(process_file, tasks, chunksize=50):
            all_results.extend(result)

    # 누적 구조
    group_sum = defaultdict(lambda: [0.0, 0])
    weekday_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0])))
    timegroup_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0])))
    total_sum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    for (stdid, node_id, ord_target, group, wd, tg), elapsed in all_results:
        group_sum[(stdid, node_id, ord_target, group)][0] += elapsed
        group_sum[(stdid, node_id, ord_target, group)][1] += 1
        weekday_sum[stdid][node_id][ord_target][wd][0] += elapsed
        weekday_sum[stdid][node_id][ord_target][wd][1] += 1
        timegroup_sum[stdid][node_id][ord_target][tg][0] += elapsed
        timegroup_sum[stdid][node_id][ord_target][tg][1] += 1
        total_sum[stdid][node_id][ord_target][0] += elapsed
        total_sum[stdid][node_id][ord_target][1] += 1

    # append 모드 처리
    if MODE == "append":
        prev_date = (datetime.strptime(TARGET_DATE, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
        prev_path = os.path.join(SAVE_DIR, f"{prev_date}.json")
        if os.path.exists(prev_path):
            with open(prev_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            for stdid, node_dict in prev_data.items():
                for node_id, ord_dict in node_dict.items():
                    for ord_key, group_dict in ord_dict.items():
                        ord_val = int(ord_key)
                        for group_key, val in group_dict.items():
                            if not group_key.startswith("wd_tg_"):
                                continue
                            s, n = val["mean"] * val["num"], val["num"]
                            group_sum[(stdid, int(node_id), ord_val, group_key)][0] += s
                            group_sum[(stdid, int(node_id), ord_val, group_key)][1] += n
                            _, _, wd, tg = group_key.split("_")
                            weekday_sum[stdid][int(node_id)][ord_val][wd][0] += s
                            weekday_sum[stdid][int(node_id)][ord_val][wd][1] += n
                            timegroup_sum[stdid][int(node_id)][ord_val][tg][0] += s
                            timegroup_sum[stdid][int(node_id)][ord_val][tg][1] += n
                            total_sum[stdid][int(node_id)][ord_val][0] += s
                            total_sum[stdid][int(node_id)][ord_val][1] += n

    # 평균 계산 및 저장
    mean_node = defaultdict(lambda: defaultdict(dict))
    for (stdid, node_id, ord_val, group), (s, n) in group_sum.items():
        mean_node[stdid][str(node_id)].setdefault(str(ord_val), {})[group] = {"mean": s / n, "num": n}

    for stdid in weekday_sum:
        for node_id in weekday_sum[stdid]:
            for ord_val in weekday_sum[stdid][node_id]:
                for wd, (s, n) in weekday_sum[stdid][node_id][ord_val].items():
                    mean_node[stdid][str(node_id)].setdefault(str(ord_val), {})[f"weekday_{wd}"] = {"mean": s / n, "num": n}

    for stdid in timegroup_sum:
        for node_id in timegroup_sum[stdid]:
            for ord_val in timegroup_sum[stdid][node_id]:
                for tg, (s, n) in timegroup_sum[stdid][node_id][ord_val].items():
                    mean_node[stdid][str(node_id)].setdefault(str(ord_val), {})[f"timegroup_{tg}"] = {"mean": s / n, "num": n}

    for stdid in total_sum:
        for node_id in total_sum[stdid]:
            for ord_val, (s, n) in total_sum[stdid][node_id].items():
                mean_node[stdid][str(node_id)].setdefault(str(ord_val), {})["total"] = {"mean": s / n, "num": n}

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(mean_node, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {SAVE_PATH}")