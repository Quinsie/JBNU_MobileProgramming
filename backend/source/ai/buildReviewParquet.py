# backend/source/ai/buildReviewParquet.py

import os
import sys
import json
import time
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

INFER_PARQUET_DIR = os.path.join(BASE_DIR, "data", "preprocess", "first_train", "inference")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
SELF_REVIEW_SAVE_DIR = os.path.join(BASE_DIR, "data", "preprocess", "first_train", "self_review")
os.makedirs(SELF_REVIEW_SAVE_DIR, exist_ok=True)

def process_file(args):
    stdid, fname, target_date = args
    result = {}
    stdid_path = os.path.join(RAW_DIR, stdid)
    hhmm = fname.replace(".json", "").split("_")[-1]
    trip_group_id = f"{target_date}_{hhmm}_{stdid}"

    with open(os.path.join(stdid_path, fname), encoding='utf-8') as f:
        data = json.load(f).get("stop_reached_logs", [])
    if len(data) < 2:
        return result

    base_time = datetime.strptime(data[0]['time'], "%Y-%m-%d %H:%M:%S")
    for record in data[1:]:
        ord = record['ord']
        arr_time = datetime.strptime(record['time'], "%Y-%m-%d %H:%M:%S")
        real_elapsed = (arr_time - base_time).total_seconds()
        result[(trip_group_id, ord)] = real_elapsed
    return result

def load_all_raw_logs_parallel(target_date):
    tasks = []
    for stdid in os.listdir(RAW_DIR):
        stdid_path = os.path.join(RAW_DIR, stdid)
        if not os.path.isdir(stdid_path):
            continue
        for fname in os.listdir(stdid_path):
            if not fname.startswith(target_date) or not fname.endswith(".json"):
                continue
            tasks.append((stdid, fname, target_date))

    merged_result = {}
    with Pool(cpu_count()) as pool:
        results = pool.map(process_file, tasks)
        for r in results:
            merged_result.update(r)
    return merged_result

def build_review_parquet(date_str):
    print(f"[INFO] Start buildReviewParquet: {date_str}")
    prev_date_str = (datetime.strptime(date_str, "%Y%m%d") - pd.Timedelta(days=1)).strftime("%Y%m%d")

    infer_path = os.path.join(INFER_PARQUET_DIR, f"{date_str}.parquet")
    if not os.path.exists(infer_path):
        raise FileNotFoundError(f"[ERROR] Inference parquet not found: {infer_path}")
    df = pd.read_parquet(infer_path)

    log_lookup = load_all_raw_logs_parallel(prev_date_str)

    matched_y = []
    matched_idx = []
    for idx, row in df.iterrows():
        key = (row['trip_group_id'], row['ord'])
        if key in log_lookup:
            matched_y.append(log_lookup[key])
            matched_idx.append(idx)

    df_matched = df.loc[matched_idx].copy()
    df_matched['y'] = [min(max(y, 0), 7200) for y in matched_y]  # clamp
    save_path = os.path.join(SELF_REVIEW_SAVE_DIR, f"{date_str}.parquet")
    df_matched.to_parquet(save_path, index=False)
    print(f"[INFO] Saved review parquet: {save_path}, rows={len(df_matched)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="대상 날짜 (YYYYMMDD)")
    args = parser.parse_args()

    now = time.time()
    build_review_parquet(args.date)
    print("소요 시간: ", time.time() - now, "sec")