# backend/source/helpers/analyzeSecondETAError.py

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

REVIEW_JSON_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "second_train", "self_review")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "analysis", "second_model")
os.makedirs(SAVE_DIR, exist_ok=True)

# === 지표 계산 함수 ===
def calculate_metrics(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    error = np.abs(pred - true)

    mae = float(np.mean(error))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mape = float(np.mean(error / (true + 1e-6))) * 100
    max_error = float(np.max(error))
    p25 = float(np.percentile(error, 25))
    p50 = float(np.percentile(error, 50))
    p75 = float(np.percentile(error, 75))
    p95 = float(np.percentile(error, 95))
    sde = float(np.std(error))
    over_60 = float(np.mean(error > 60)) * 100  # 1분 초과 비율 %

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2),
        "max_error": round(max_error, 3),
        "p25": round(p25, 3),
        "p50": round(p50, 3),
        "p75": round(p75, 3),
        "p95": round(p95, 3),
        "sde": round(sde, 3),
        "over_60_sec(%)": round(over_60, 2)
    }

# === 그룹별 집계 함수 ===
def evaluate_group(df, groupby_key):
    result = {}
    for key, group in df.groupby(groupby_key, observed=True):
        stat = {}
        for i in range(1, 6):
            pred = group[f"x_prev_pred_elapsed_{i}"] * 3000
            true = group[f"y_{i}"] * 3000
            stat[f"ORD+{i}"] = calculate_metrics(pred, true)
        result[str(key)] = stat
    return result

# === 메인 분석 함수 ===
def analyze_second_eta(date_str):
    print(f"[INFO] 2차 ETA 분석 시작: {date_str}")
    
    json_path = os.path.join(REVIEW_JSON_DIR, f"{date_str}.json")
    save_path = os.path.join(SAVE_DIR, f"{date_str}.json")

    if not os.path.exists(json_path):
        print(f"[ERROR] 파일 없음: {json_path}")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # 정규화 해제용 열 생성
    for i in range(1, 6):
        df[f"pred_{i}"] = df[f"x_prev_pred_elapsed_{i}"] * 3000
        df[f"true_{i}"] = df[f"y_{i}"] * 3000
        df[f"abs_error_{i}"] = np.abs(df[f"pred_{i}"] - df[f"true_{i}"])

    # ord_ratio group 구간
    df["ord_ratio_group"] = pd.cut(df["x_node_id_ratio"], bins=[0, 0.25, 0.5, 0.75, 1.0], labels=["0_25", "26_50", "51_75", "76_100"])

    result = {
        "overall": evaluate_group(df, lambda x: "overall"),
        "ord": evaluate_group(df, "ord"),
        "weekday": evaluate_group(df, "x_weekday"),
        "timegroup": evaluate_group(df, "x_timegroup"),
        "wd_tg": evaluate_group(df, "x_weekday_timegroup"),
        "stdid": evaluate_group(df, df["trip_group_id"].str.split("_").str[-1]),  # stdid
        "bus_number": evaluate_group(df, df["x_bus_number"].astype(str)),  # numeric code
        "ord_ratio": evaluate_group(df, "ord_ratio_group")
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] 분석 결과 저장 완료: {save_path}")

# === CLI 진입 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYYMMDD 날짜 형식")
    args = parser.parse_args()

    analyze_second_eta(args.date)