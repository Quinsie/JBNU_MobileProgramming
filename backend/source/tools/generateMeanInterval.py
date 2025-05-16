# backend/source/tools/generateMeanInterval.py

import os
import sys
import json
from datetime import datetime, timedelta
from collections import defaultdict

# 날짜 설정
TARGET_DATE = "20250505"
MODE = "init"  # or "append"

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
ELAPSED_PATH = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed", f"{TARGET_DATE}.json")
STOP_INDEX_PATH = os.path.join(BASE_DIR, "data", "processed", "stop_to_routes.json")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "mean", "interval", f"{TARGET_DATE}.json")
PREV_PATH = os.path.join(BASE_DIR, "data", "processed", "mean", "interval", f"{(datetime.strptime(TARGET_DATE, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')}.json")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# 데이터 로딩
with open(ELAPSED_PATH, "r", encoding="utf-8") as f:
    mean_elapsed = json.load(f)
with open(STOP_INDEX_PATH, "r", encoding="utf-8") as f:
    stop_index = json.load(f)

# 전일 mean_interval 누적용
if MODE == "append" and os.path.exists(PREV_PATH):
    with open(PREV_PATH, "r", encoding="utf-8") as f:
        prev_data = json.load(f)
else:
    prev_data = {}

# 누적 구조 초기화
interval_sum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # stop_id → group → [sum, num]

# 계산 시작
for stop_id, route_list in stop_index.items():
    for entry in route_list:
        stdid = entry.get("stdid")
        ord_val = entry.get("ord")
        if not stdid or not ord_val or ord_val <= 1:
            continue

        ord_str = str(ord_val)
        ord_prev_str = str(ord_val - 1)

        if stdid not in mean_elapsed:
            continue
        if ord_str not in mean_elapsed[stdid] or ord_prev_str not in mean_elapsed[stdid]:
            continue

        for group in mean_elapsed[stdid][ord_str]:
            if group not in mean_elapsed[stdid][ord_prev_str]:
                continue
            v1 = mean_elapsed[stdid][ord_str][group]
            v0 = mean_elapsed[stdid][ord_prev_str][group]

            if v1["num"] > 0 and v0["num"] > 0:
                diff = v1["mean"] - v0["mean"]
                if diff >= 0:
                    interval_sum[stop_id][group][0] += diff
                    interval_sum[stop_id][group][1] += 1

# append 모드 누적 반영
for stop_id in prev_data:
    for group, val in prev_data[stop_id].items():
        interval_sum[stop_id][group][0] += val["mean"] * val["num"]
        interval_sum[stop_id][group][1] += val["num"]

# 결과 정리 및 저장
mean_interval = defaultdict(dict)
for stop_id in interval_sum:
    for group, (s, n) in interval_sum[stop_id].items():
        if n > 0:
            mean_interval[stop_id][group] = {"mean": s / n, "num": n}

# weekday_* 계산
for stop_id in mean_interval:
    weekday_sum = {"1": [0.0, 0], "2": [0.0, 0], "3": [0.0, 0]}
    for group in mean_interval[stop_id]:
        if group.startswith("wd_tg_"):
            parts = group.split("_")
            wd = parts[2]  # weekday number
            s = mean_interval[stop_id][group]["mean"] * mean_interval[stop_id][group]["num"]
            n = mean_interval[stop_id][group]["num"]
            weekday_sum[wd][0] += s
            weekday_sum[wd][1] += n
    for wd, (s, n) in weekday_sum.items():
        if n > 0:
            mean_interval[stop_id][f"weekday_{wd}"] = {"mean": s / n, "num": n}

    # total
    total_s, total_n = 0.0, 0
    for wd in ["1", "2", "3"]:
        total_s += weekday_sum[wd][0]
        total_n += weekday_sum[wd][1]
    if total_n > 0:
        mean_interval[stop_id]["total"] = {"mean": total_s / total_n, "num": total_n}

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(mean_interval, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {SAVE_PATH}")