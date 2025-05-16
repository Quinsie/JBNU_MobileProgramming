# backend/source/tools/generateMeanInterval.py

import os
import sys
import json
from datetime import datetime, timedelta
from collections import defaultdict

# 날짜 설정
# TARGET_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
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
interval_sum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
weekday_sum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
timegroup_sum = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
total_sum = defaultdict(lambda: [0.0, 0])

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

                    if group.startswith("wd_tg_"):
                        _, _, wd, tg = group.split("_")
                        weekday_sum[stop_id][wd][0] += diff
                        weekday_sum[stop_id][wd][1] += 1
                        timegroup_sum[stop_id][tg][0] += diff
                        timegroup_sum[stop_id][tg][1] += 1
                        total_sum[stop_id][0] += diff
                        total_sum[stop_id][1] += 1

# append 모드 누적 반영
for stop_id in prev_data:
    for group, val in prev_data[stop_id].items():
        s = val["mean"] * val["num"]
        n = val["num"]
        interval_sum[stop_id][group][0] += s
        interval_sum[stop_id][group][1] += n

        if group.startswith("wd_tg_"):
            try:
                _, _, wd, tg = group.split("_")
            except:
                continue
            weekday_sum[stop_id][wd][0] += s
            weekday_sum[stop_id][wd][1] += n
            timegroup_sum[stop_id][tg][0] += s
            timegroup_sum[stop_id][tg][1] += n
            total_sum[stop_id][0] += s
            total_sum[stop_id][1] += n

# 결과 정리
mean_interval = defaultdict(dict)
for stop_id in interval_sum:
    for group, (s, n) in interval_sum[stop_id].items():
        if n > 0:
            mean_interval[stop_id][group] = {"mean": s / n, "num": n}

for stop_id in weekday_sum:
    for wd_key, (s, n) in weekday_sum[stop_id].items():
        if n > 0:
            mean_interval[stop_id][f"weekday_{wd_key}"] = {"mean": s / n, "num": n}

for stop_id in timegroup_sum:
    for tg_key, (s, n) in timegroup_sum[stop_id].items():
        if n > 0:
            mean_interval[stop_id][f"timegroup_{tg_key}"] = {"mean": s / n, "num": n}

for stop_id, (s, n) in total_sum.items():
    if n > 0:
        mean_interval[stop_id]["total"] = {"mean": s / n, "num": n}

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(mean_interval, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {SAVE_PATH}")