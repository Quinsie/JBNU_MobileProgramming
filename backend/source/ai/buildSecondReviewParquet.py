# backend/source/ai/buildSecondReviewParquet.py

import os
import json
import time
import argparse
import pandas as pd
from datetime import datetime, timedelta

# === BASE 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REVIEW_JSON_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "second_train", "review")
REVIEW_PARQUET_DIR = REVIEW_JSON_DIR  # 같은 경로에 저장

def convert_json_to_parquet(target_date):
    print("[INFO] Review Parquet 제작 시작...")
    json_path = os.path.join(REVIEW_JSON_DIR, f"{target_date}.json")
    parquet_path = os.path.join(REVIEW_PARQUET_DIR, f"{target_date}.parquet")

    if not os.path.exists(json_path):
        print(f"[ERROR] JSON 파일이 존재하지 않습니다: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    print(f"[INFO] 저장 완료: {parquet_path}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="대상 날짜 (YYYYmmDD 형식)")
    args = parser.parse_args()

    now = time.time()
    target_date = datetime.strptime(args.date, "%Y%m%d") - timedelta(days=1)
    convert_json_to_parquet(datetime.strftime(target_date, "%Y%m%d"))
    print("소요 시간: ", round(time.time() - now, 1), "sec")