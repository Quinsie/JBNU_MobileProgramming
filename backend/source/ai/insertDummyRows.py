# backend/source/ai/insertDummyRows.py

import pandas as pd
import os

# === 설정 ===
DATE = "20250506"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "replay", f"{DATE}.parquet")

def insert_dummy_rows():
    if not os.path.exists(PARQUET_PATH):
        print(f"[ERROR] {PARQUET_PATH} 파일이 존재하지 않습니다.")
        return

    # 기존 parquet 불러오기
    df = pd.read_parquet(PARQUET_PATH)
    print(f"[INFO] 기존 row 수: {len(df)}")

    # 샘플 행 하나 복사
    sample = df.sample(1).iloc[0]
    dummy_rows = []

    # 평일(1), 토요일(2) 각각 timegroup 1~8 조합
    for weekday in [1, 2]:
        for tg in range(1, 9):
            wdtg = weekday * 8 + (tg - 1)
            new_row = sample.copy()

            # 핵심 필드만 바꿔서 dummy로 처리
            new_row["trip_group_id"] = f"dummy_{weekday}_{tg}"
            new_row["ord"] = 1
            new_row["y"] = 0.0

            new_row["x_weekday"] = weekday
            new_row["x_timegroup"] = tg
            new_row["x_weekday_timegroup"] = wdtg

            dummy_rows.append(new_row)

    # 병합 및 저장
    df_augmented = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)
    df_augmented.to_parquet(PARQUET_PATH, index=False)

    print(f"[INFO] dummy row 16개 추가 완료. 총 row 수: {len(df_augmented)}")

if __name__ == "__main__":
    insert_dummy_rows()