# backend/source/ai/insertDummyRows.py

import pandas as pd
import os
import math
from datetime import datetime, timedelta

# === 설정 ===
DATE = "20250507"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "replay", f"{DATE}.parquet")

def time_to_sin_cos(dt):
    minutes = dt.hour * 60 + dt.minute
    angle = 2 * math.pi * (minutes / 1440)
    return math.sin(angle), math.cos(angle)

def insert_dummy_rows():
    if not os.path.exists(PARQUET_PATH):
        print(f"[ERROR] {PARQUET_PATH} 파일이 존재하지 않습니다.")
        return

    # 기존 parquet 불러오기
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[INFO] 기존 row 수: {len(df)}")

    # 샘플 320개 무작위 추출
    dummy_source = df.sample(320).reset_index(drop=True)
    dummy_rows = []

    index = 0
    for weekday in [1, 2]:
        for tg in range(1, 9):
            wdtg = weekday * 8 + (tg - 1)
            for _ in range(20):
                row = dummy_source.iloc[index].copy()
                index += 1

                # trip_group_id 수정
                row["trip_group_id"] = f"dummy_{weekday}_{tg}_{index}"
                row["ord"] = 1

                # 날짜/시간 고정: 05:00 + wdtg * 5min
                dummy_time = datetime(2025, 5, 7, 5, 0) + timedelta(minutes=wdtg * 5)
                sin_val, cos_val = time_to_sin_cos(dummy_time)

                row["x_weekday"] = weekday
                row["x_timegroup"] = tg
                row["x_weekday_timegroup"] = wdtg
                row["x_departure_time_sin"] = sin_val
                row["x_departure_time_cos"] = cos_val

                # 레이블도 샘플 그대로 유지
                # row["y"] 그대로 유지

                dummy_rows.append(row)

    df_augmented = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)
    df_augmented.to_parquet(PARQUET_PATH, index=False)
    print(f"[INFO] dummy row {len(dummy_rows)}개 추가 완료. 총 row 수: {len(df_augmented)}")

if __name__ == "__main__":
    insert_dummy_rows()