# backend/source/helpers/analyzeETAError.py

import os
import sys
import json
import math
import argparse
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# ==== 경로 설정 ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

ETA_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", "first_model")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
MEAN_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
STDID_MAP_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "analysis", "first_model")
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== 지표 계산 함수 ====
def calculate_metrics(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    error = np.abs(pred - true)
    
    mae = float(np.mean(error))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mape = float(np.mean(error / (true + 1e-6))) * 100  # avoid div0
    max_error = float(np.max(error))
    p25 = float(np.percentile(error, 25))
    p50 = float(np.percentile(error, 50))
    p75 = float(np.percentile(error, 75))
    p95 = float(np.percentile(error, 95))
    sde = float(np.std(error))

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2),
        "max_error": round(max_error, 3),
        "p25": round(p25, 3),
        "p50": round(p50, 3),
        "p75": round(p75, 3),
        "p95": round(p95, 3),
        "sde": round(sde, 3)
    }

# ==== 메인 분석 함수 ====
def analyze_eta(date_str):
    print(f"[INFO] ETA 분석 시작: {date_str}")

    eta_path = os.path.join(ETA_DIR, f"{date_str}.json")
    save_path = os.path.join(SAVE_DIR, f"{date_str}.json")
    mean_path = os.path.join(MEAN_DIR, f"{date_str}.json")
    stdid_map = json.load(open(STDID_MAP_PATH, encoding='utf-8'))
    eta_table = json.load(open(eta_path, encoding='utf-8'))
    mean_elapsed = json.load(open(mean_path, encoding='utf-8'))

    # 분석용 구조: {category: {'pred': [], 'true': [], 'mean': []}}
    stats = defaultdict(lambda: defaultdict(list))

    for key, ord_dict in eta_table.items():
        stdid, hhmm = key.split("_")
        route_name = stdid_map.get(stdid, stdid)
        bus_number = route_name[:-2]  # e.g., "10" from "10A1"
        fname = f"{date_str}_{hhmm}.json"
        raw_path = os.path.join(RAW_DIR, stdid, fname)
        if not os.path.exists(raw_path): continue

        with open(raw_path, encoding='utf-8') as f:
            log = json.load(f).get("stop_reached_logs", [])

        if len(log) < 2: continue
        base_time = datetime.strptime(log[0]['time'], "%Y-%m-%d %H:%M:%S")
        weekday = base_time.weekday()
        hour = base_time.hour
        tg = 1 + (hour * 60 + base_time.minute - 330) // 90
        tg = tg if 1 <= tg <= 8 else 8
        wd_tg = f"{weekday}_{tg}"

        for record in log[1:]:
            ord = str(record['ord'])
            pred_str = ord_dict.get(ord)
            if pred_str is None: continue

            pred_time = datetime.strptime(pred_str, "%Y-%m-%d %H:%M:%S")
            true_time = datetime.strptime(record['time'], "%Y-%m-%d %H:%M:%S")
            
            # 날짜 넘어간 예측으로 간주해서 하루 더해줌
            if pred_time < base_time: pred_time += timedelta(days=1)
            if true_time < base_time: true_time += timedelta(days=1)

            elapsed_pred = (pred_time - base_time).total_seconds()
            elapsed_true = (true_time - base_time).total_seconds()

            mean_val = mean_elapsed.get(stdid, {}).get(ord, {}).get(f"wd_tg_{wd_tg}", {}).get("mean", None)
            elapsed_mean = mean_val if mean_val is not None else 0.0

            try:
                ord_num = int(ord)
                max_ord = int(log[-1]['ord'])
                ord_ratio = ord_num / max_ord
            except:
                ord_ratio = 0.0  # fallback

            if ord_ratio <= 0.25:
                ord_group = "ord_0_25"
            elif ord_ratio <= 0.5:
                ord_group = "ord_26_50"
            elif ord_ratio <= 0.75:
                ord_group = "ord_51_75"
            else:
                ord_group = "ord_76_100"

            for cat in ["overall", f"weekday_{weekday}", f"tg_{tg}", f"wdtg_{wd_tg}", f"route_{route_name}", f"bus_{bus_number}", ord_group]:
                stats[cat]['pred'].append(elapsed_pred)
                stats[cat]['true'].append(elapsed_true)
                stats[cat]['mean'].append(elapsed_mean)
                stats[cat]['errors'].append({
                    "stdid": stdid,
                    "route_name": route_name,
                    "bus_number": bus_number,
                    "hhmm": hhmm,
                    "ord": ord,
                    "pred": elapsed_pred,
                    "true": elapsed_true,
                    "abs_error": abs(elapsed_pred - elapsed_true)
                })

    # 지표 계산
    result = {}
    for cat in stats:
        eta = stats[cat]['pred']
        true = stats[cat]['true']
        # mean = stats[cat]['mean']
        errors = stats[cat]['errors']

        result[cat] = {
            "pred_vs_true": calculate_metrics(eta, true),
            # "pred_vs_mean": calculate_metrics(eta, mean),
        }

        # meta 정보 추가
        over_10 = sum(1 for e in errors if e['abs_error'] > 600)
        top10 = sorted(errors, key=lambda e: e['abs_error'], reverse=True)[:10]

        result[cat]["_meta"] = {
            "over_10min_errors": over_10,
            "top10_errors": top10
        }

    # 저장
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] 저장 완료: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    analyze_eta(args.date)