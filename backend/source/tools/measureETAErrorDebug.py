import os
import json
import numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ETA_TABLE_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table")

# 날짜 입력
target_date = input("ETA Table 날짜를 입력하세요 (YYYYMMDD): ").strip()
eta_path = os.path.join(ETA_TABLE_DIR, f"{target_date}.json")

# 존재 확인
if not os.path.exists(eta_path):
    print(f"[ERROR] 파일이 존재하지 않습니다: {eta_path}")
    exit()

# ETA Table 로드
with open(eta_path, "r", encoding="utf-8") as f:
    eta_table = json.load(f)

err_list = []

# 오차 수집
for stdid in eta_table:
    for record in eta_table[stdid]:
        eta = record.get("ETA")
        real = record.get("REAL")
        if eta is None or real is None:
            continue
        err = real - eta  # 초 단위
        err_list.append(err)

# 분석
err_array = np.array(err_list)
total = len(err_array)
over_600 = np.sum(np.abs(err_array) > 600)
over_1800 = np.sum(np.abs(err_array) > 1800)
over_3600 = np.sum(np.abs(err_array) > 3600)

print(f"\n총 비교 수: {total}")
print(f"평균 오차 (초): {np.mean(err_array):.2f}")
print(f"표준편차 (초): {np.std(err_array):.2f}")
print(f"최대 오차 (초): {np.max(np.abs(err_array))}")
print()
print(f"10분 이상 오차 비율: {over_600 / total * 100:.2f}%")
print(f"30분 이상 오차 비율: {over_1800 / total * 100:.2f}%")
print(f"1시간 이상 오차 비율: {over_3600 / total * 100:.2f}%")