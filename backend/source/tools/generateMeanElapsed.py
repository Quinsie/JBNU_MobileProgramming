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
PREV_PATH = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed", (datetime.strptime(TARGET_DATE, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d") + ".json")
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

    wd = str(get_weekday_type(departure_time))
    tg = str(get_time_group(departure_time))
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

# main
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

    # 누적용 구조
    group_sum = defaultdict(lambda: [0.0, 0])
    mean_elapsed = defaultdict(lambda: defaultdict(dict))

    # 누적
    for (stdid, ord_val, group, wd, tg), elapsed in all_results:
        group_sum[(stdid, ord_val, group)] = [
            group_sum[(stdid, ord_val, group)][0] + elapsed,
            group_sum[(stdid, ord_val, group)][1] + 1
        ]

    # 기존 mean 불러와서 누적 (append 모드)
    if MODE == "append" and os.path.exists(PREV_PATH):
        with open(PREV_PATH, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
        for stdid, ord_dict in prev_data.items():
            for ord_str, group_dict in ord_dict.items():
                ord_val = int(ord_str)
                for group_key, val in group_dict.items():
                    if not group_key.startswith("wd_tg_"):
                        continue
                    s, n = val["mean"] * val["num"], val["num"]
                    group_sum[(stdid, ord_val, group_key)][0] += s
                    group_sum[(stdid, ord_val, group_key)][1] += n

    # wd_tg 저장
    for (stdid, ord_val, group), (s, n) in group_sum.items():
        mean_elapsed[stdid][ord_val][group] = {"mean": s / n, "num": n}

    # weekday_* 계산
    for stdid in mean_elapsed:
        for ord_val in mean_elapsed[stdid]:
            weekday_sums = defaultdict(lambda: [0.0, 0])
            for group in mean_elapsed[stdid][ord_val]:
                if group.startswith("wd_tg_"):
                    _, _, wd, tg = group.split("_")
                    weekday_sums[wd][0] += mean_elapsed[stdid][ord_val][group]["mean"] * mean_elapsed[stdid][ord_val][group]["num"]
                    weekday_sums[wd][1] += mean_elapsed[stdid][ord_val][group]["num"]
            for wd in weekday_sums:
                s, n = weekday_sums[wd]
                mean_elapsed[stdid][ord_val][f"weekday_{wd}"] = {"mean": s / n, "num": n}

    # timegroup_* 계산
    for stdid in mean_elapsed:
        for ord_val in mean_elapsed[stdid]:
            tg_sums = defaultdict(lambda: [0.0, 0])
            for group in mean_elapsed[stdid][ord_val]:
                if group.startswith("wd_tg_"):
                    _, _, wd, tg = group.split("_")
                    tg_sums[tg][0] += mean_elapsed[stdid][ord_val][group]["mean"] * mean_elapsed[stdid][ord_val][group]["num"]
                    tg_sums[tg][1] += mean_elapsed[stdid][ord_val][group]["num"]
            for tg in tg_sums:
                s, n = tg_sums[tg]
                mean_elapsed[stdid][ord_val][f"timegroup_{tg}"] = {"mean": s / n, "num": n}

    # total 계산
    for stdid in mean_elapsed:
        for ord_val in mean_elapsed[stdid]:
            s_total, n_total = 0.0, 0
            for group in mean_elapsed[stdid][ord_val]:
                if group.startswith("wd_tg_"):
                    v = mean_elapsed[stdid][ord_val][group]
                    s_total += v["mean"] * v["num"]
                    n_total += v["num"]
            if n_total > 0:
                mean_elapsed[stdid][ord_val]["total"] = {"mean": s_total / n_total, "num": n_total}

    # 저장
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(mean_elapsed, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {SAVE_PATH}")