# backend/source/ai/insertDummyRows.py

import pandas as pd
import os

# === 설정 ===
DATE = "20250507"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "replay", f"{DATE}.parquet")

# dummy 개수 설정
DUMMY_PER_GROUP = 20  # weekday × timegroup 조합당 개수 (총 320개)

def insert_dummy_rows():
    if not os.path.exists(PARQUET_PATH):
        print(f"[ERROR] {PARQUET_PATH} 파일이 존재하지 않습니다.")
        return

    # 기존 parquet 불러오기
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[INFO] 기존 row 수: {len(df)}")

    # 다양한 sample 확보
    total_dummies = DUMMY_PER_GROUP * 2 * 8
    samples = df.sample(n=total_dummies, replace=True).reset_index(drop=True)
    sample_idx = 0

    dummy_rows = []

    # weekday (1, 2) × timegroup (1~8)
    for weekday in [1, 2]:
        for tg in range(1, 9):
            wdtg = weekday * 8 + (tg - 1)

            for _ in range(DUMMY_PER_GROUP):
                new_row = samples.iloc[sample_idx].copy()
                sample_idx += 1

                new_row["trip_group_id"] = f"dummy_{weekday}_{tg}_{sample_idx}"
                new_row["ord"] = 1
                new_row["y"] = 0.0
                new_row["x_weekday"] = weekday
                new_row["x_timegroup"] = tg
                new_row["x_weekday_timegroup"] = wdtg

                dummy_rows.append(new_row)

    # 병합 및 저장
    df_augmented = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)
    df_augmented.to_parquet(PARQUET_PATH, index=False)

    print(f"[INFO] dummy row {total_dummies}개 추가 완료. 총 row 수: {len(df_augmented)}")

if __name__ == "__main__":
    insert_dummy_rows()