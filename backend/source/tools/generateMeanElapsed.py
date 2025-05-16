# backend/source/tools/generateMeanElapsed.py

import os
import sys
import json
from datetime import datetime, timedelta
from collections import defaultdict
from multiprocessing import Pool

# 날짜 설정 (YYYYMMDD)
# TARGET_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
TARGET_DATE = "20250505"
MODE = "init"  # "init" or "append"

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed", f"{TARGET_DATE}.json")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
from source.utils.getDayType import getDayType

# 시간대 그룹핑 (8단계)
def get_time_group(departure_time):
    minutes = departure_time.hour * 60 + departure_time.minute
    if 330 <= minutes < 420:
        return 0
    elif 420 <= minutes < 540:
        return 1
    elif 540 <= minutes < 690:
        return 2
    elif 690 <= minutes < 840:
        return 3
    elif 840 <= minutes < 1020:
        return 4
    elif 1020 <= minutes < 1140:
        return 5
    elif 1140 <= minutes < 1260:
        return 6
    else:
        return 7

# 요일 그룹핑: 1(평일), 2(토), 3(공휴일)
def get_weekday_type(departure_time):
    day_type = getDayType(departure_time)  # 'weekday' | 'saturday' | 'holiday'
    return {'weekday': 1, 'saturday': 2, 'holiday': 3}[day_type]

# 파일 1개 처리
def process_file(args):
    stdid, file_path = args
    filename = os.path.basename(file_path)
    time_str = filename.replace(".json", "").split("_")[-1]  # HHMM
    departure_time = datetime.strptime(f"{TARGET_DATE}_{time_str}", "%Y%m%d_%H%M")

    wd = get_weekday_type(departure_time)
    tg = get_time_group(departure_time) + 1
    group = f"wd_tg_{wd}_{tg}"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logs = data.get("stop_reached_logs", [])

    result = []
    if not logs or len(logs) < 2:
        return result

    try:
        base_time = datetime.strptime(logs[0]["time"], "%Y-%m-%d %H:%M:%S")
    except:
        return result

    for item in logs[1:]:
        ord_val = item.get("ord")
        if ord_val is None:
            continue
        try:
            arr_time = datetime.strptime(item["time"], "%Y-%m-%d %H:%M:%S")
        except:
            continue
        elapsed = (arr_time - base_time).total_seconds()
        result.append(((stdid, int(ord_val), group, wd, tg), elapsed))

    return result

# 메인
if __name__ == "__main__":
    tasks = []
    for stdid in os.listdir(RAW_DIR):
        std_path = os.path.join(RAW_DIR, stdid)
        if not os.path.isdir(std_path):
            continue
        for fname in os.listdir(std_path):
            if not fname.startswith(TARGET_DATE):
                continue
            fpath = os.path.join(std_path, fname)
            tasks.append((stdid, fpath))

    all_results = []
    with Pool() as pool:
        for result in pool.imap_unordered(process_file, tasks, chunksize=50):
            all_results.extend(result)

    group_sum = defaultdict(lambda: [0.0, 0])  # (stdid, ord, group) -> [sum, count]
    weekday_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0])))  # [stdid][ord][wd]
    timegroup_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0])))  # [stdid][ord][tg]
    total_sum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

    for (stdid, ord_val, group, wd, tg), elapsed in all_results:
        group_sum[(stdid, ord_val, group)][0] += elapsed
        group_sum[(stdid, ord_val, group)][1] += 1

        weekday_sum[stdid][ord_val][wd][0] += elapsed
        weekday_sum[stdid][ord_val][wd][1] += 1

        timegroup_sum[stdid][ord_val][tg][0] += elapsed
        timegroup_sum[stdid][ord_val][tg][1] += 1

        total_sum[stdid][ord_val][0] += elapsed
        total_sum[stdid][ord_val][1] += 1

    mean_elapsed = defaultdict(lambda: defaultdict(dict))

    for (stdid, ord_val, group), (s, n) in group_sum.items():
        mean_elapsed[stdid][ord_val][group] = {"mean": s / n, "num": n}

    for stdid in weekday_sum:
        for ord_val in weekday_sum[stdid]:
            for wd, (s, n) in weekday_sum[stdid][ord_val].items():
                mean_elapsed[stdid][ord_val][f"weekday_{wd}"] = {"mean": s / n, "num": n}

    for stdid in timegroup_sum:
        for ord_val in timegroup_sum[stdid]:
            for tg, (s, n) in timegroup_sum[stdid][ord_val].items():
                mean_elapsed[stdid][ord_val][f"timegroup_{tg}"] = {"mean": s / n, "num": n}

    for stdid in total_sum:
        for ord_val, (s, n) in total_sum[stdid].items():
            mean_elapsed[stdid][ord_val]["total"] = {"mean": s / n, "num": n}

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(mean_elapsed, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {SAVE_PATH}")