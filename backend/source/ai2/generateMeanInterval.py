# backend/source/ai2/generateMeanInterval.py
# 1차 모델 확장 실험

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from collections import defaultdict

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="날짜 (YYYYMMDD)")
    parser.add_argument("--mode", required=True, choices=["init", "append"], help="모드 (init 또는 append)")
    args = parser.parse_args()

    TARGET_DATE = args.date
    MODE = args.mode

    ELAPSED_PATH = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed", f"{TARGET_DATE}_2.json")
    STOP_INDEX_PATH = os.path.join(BASE_DIR, "data", "processed", "stop_to_routes.json")
    SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "mean", "interval", f"{TARGET_DATE}_2.json")
    PREV_PATH = os.path.join(BASE_DIR, "data", "processed", "mean", "interval", f"{(datetime.strptime(TARGET_DATE, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')}_2.json")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # 데이터 로드
    with open(ELAPSED_PATH, "r", encoding="utf-8") as f:
        mean_elapsed = json.load(f)

    with open(STOP_INDEX_PATH, "r", encoding="utf-8") as f:
        stop_index = json.load(f)

    # append 모드라면 이전 평균도 로드
    prev_data = {}
    if MODE == "append" and os.path.exists(PREV_PATH):
        with open(PREV_PATH, "r", encoding="utf-8") as f:
            prev_data = json.load(f)

    # 누적용 딕셔너리
    time_sum = defaultdict(lambda: [0.0, 0])
    group_sum = defaultdict(lambda: [0.0, 0])
    weekday_sum = defaultdict(lambda: [0.0, 0])
    timegroup_sum = defaultdict(lambda: [0.0, 0])
    total_sum = defaultdict(lambda: [0.0, 0])

    # 계산
    for stop_id, route_list in stop_index.items():
        for entry in route_list:
            stdid = entry["stdid"]
            ord_val = entry["ord"]
            if ord_val <= 1:
                continue
            ord_str = str(ord_val)
            ord_prev_str = str(ord_val - 1)

            if stdid not in mean_elapsed:
                continue
            if ord_str not in mean_elapsed[stdid] or ord_prev_str not in mean_elapsed[stdid]:
                continue

            for group in mean_elapsed[stdid][ord_str]:
                if group.startswith("time_"):
                    if group not in mean_elapsed[stdid][ord_prev_str]:
                        continue
                    v1 = mean_elapsed[stdid][ord_str][group]
                    v0 = mean_elapsed[stdid][ord_prev_str][group]
                    if v1["num"] > 0 and v0["num"] > 0:
                        diff = v1["mean"] - v0["mean"]
                        if diff >= 0:
                            _, wd, hhmm = group.split("_")
                            time_sum[(stop_id, wd, hhmm)][0] += diff
                            time_sum[(stop_id, wd, hhmm)][1] += 1

                elif group.startswith("wd_tg_"):
                    if group not in mean_elapsed[stdid][ord_prev_str]:
                        continue
                    v1 = mean_elapsed[stdid][ord_str][group]
                    v0 = mean_elapsed[stdid][ord_prev_str][group]
                    if v1["num"] > 0 and v0["num"] > 0:
                        diff = v1["mean"] - v0["mean"]
                        if diff >= 0:
                            group_sum[(stop_id, group)][0] += diff
                            group_sum[(stop_id, group)][1] += 1
                            _, _, wd, tg = group.split("_")
                            weekday_sum[(stop_id, wd)][0] += diff
                            weekday_sum[(stop_id, wd)][1] += 1
                            timegroup_sum[(stop_id, tg)][0] += diff
                            timegroup_sum[(stop_id, tg)][1] += 1
                            total_sum[stop_id][0] += diff
                            total_sum[stop_id][1] += 1

    # append 누적
    for stop_id in prev_data:
        for group, val in prev_data[stop_id].items():
            s, n = val["mean"] * val["num"], val["num"]

            if group.startswith("time_"):
                _, wd, hhmm = group.split("_")
                time_sum[(stop_id, wd, hhmm)][0] += s
                time_sum[(stop_id, wd, hhmm)][1] += n

            elif group.startswith("wd_tg_"):
                group_sum[(stop_id, group)][0] += s
                group_sum[(stop_id, group)][1] += n
                _, _, wd, tg = group.split("_")
                weekday_sum[(stop_id, wd)][0] += s
                weekday_sum[(stop_id, wd)][1] += n
                timegroup_sum[(stop_id, tg)][0] += s
                timegroup_sum[(stop_id, tg)][1] += n
                total_sum[stop_id][0] += s
                total_sum[stop_id][1] += n

    # 평균 생성
    mean_interval = defaultdict(dict)

    for (stop_id, wd, hhmm), (s, n) in time_sum.items():
        mean_interval[stop_id][f"time_{wd}_{hhmm}"] = {"mean": s / n, "num": n}

    for (stop_id, group), (s, n) in group_sum.items():
        mean_interval[stop_id][group] = {"mean": s / n, "num": n}

    for (stop_id, wd), (s, n) in weekday_sum.items():
        mean_interval[stop_id][f"weekday_{wd}"] = {"mean": s / n, "num": n}

    for (stop_id, tg), (s, n) in timegroup_sum.items():
        mean_interval[stop_id][f"timegroup_{tg}"] = {"mean": s / n, "num": n}

    for stop_id, (s, n) in total_sum.items():
        mean_interval[stop_id]["total"] = {"mean": s / n, "num": n}

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(mean_interval, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {SAVE_PATH}")