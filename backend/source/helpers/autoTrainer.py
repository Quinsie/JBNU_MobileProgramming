# backend/source/helpers/autoTrainer.py

import os
import sys
import time
import subprocess
from datetime import datetime, timedelta

s = time.time()
# 시작일자, 종료일자 설정
start_date = datetime.strptime("20250513", "%Y%m%d")
end_date = datetime.strptime("20250604", "%Y%m%d")  # 종료일 포함

# loop 이전 사전 스크립트
pre_scripts = [
    ["python3", "../ai/cleanBusLogs.py", "--start", "20250603", "--end", "20250603"],
    ["python3", "autoMeanGenerator.py"],

    ["python3", "../ai/buildFirstReplayParquetStart.py", "--date", "20250512"],
    ["python3", "../ai/trainFirstETA.py", "--date", "20250512", "--mode", "replay"],
    ["python3", "../ai/generateFirstETA.py", "--date", "20250512"],

    ["python3", "../ai/buildSecondReplayParquetStart.py", "--date", "20250512"],
    ["python3", "../ai/trainSecondETA.py", "--date", "20250512", "--mode", "replay"],
]

# 실행할 스크립트 및 인자 정의
scripts = [
    ("buildFirstReviewParquet.py", ["--date", "{date}"]),
    ("trainFirstETA.py", ["--date", "{date}", "--mode", "self_review"]),
    ("buildFirstReplayParquet.py", ["--date", "{date}"]),
    ("trainFirstETA.py", ["--date", "{date}", "--mode", "replay"]),
    ("generateFirstETA.py", ["--date", "{date}"]),
]

scripts2 = [
    ("generateSecondReviewFile.py", ["--date", "{date}"]),
    ("buildSecondReviewParquet.py", ["--date", "{date}"]),
    ("trainSecondETA.py", ["--date", "{date}", "--mode", "self_review"]),
    ("buildSecondReplayParquet.py", ["--date", "{date}"]),
    ("trainSecondETA.py", ["--date", "{date}", "--mode", "replay"]),
]

post_scripts = [
    ["python3", "autoAnalyzer.py"],
    ["python3", "autoSecondAnalyzer.py"]
]

# 전처리 부분
now = time.time()
print("전처리 시작")
for cmd in pre_scripts:
    print(f"실행 중: {' '.join(cmd)}")
    subprocess.run(cmd)
print("전처리 끝, 소요 시간: ", round(time.time() - now, 1), "sec")

# 날짜별 루프
now = time.time()
print("First Model 시작")
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")
    date_minus1 = (current_date - timedelta(days=1)).strftime("%Y%m%d")
    print(f"\n==== [{date_str}] 시작 ====")

    for script, args_template in scripts:
        target_date = date_minus1 if "Mean" in script else date_str
        args = [arg.format(date=target_date) for arg in args_template]

        full_command = ["python3", f"../ai/{script}"] + args
        print(f"실행 중: {' '.join(full_command)}")
        subprocess.run(full_command)

    current_date += timedelta(days=1)
print("First Model 끝, 소요 시간: ", round(time.time() - now, 1), "sec")

# 날짜별 루프
now = time.time()
print("Second Model 시작")
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")
    date_minus1 = (current_date - timedelta(days=1)).strftime("%Y%m%d")
    print(f"\n==== [{date_str}] 시작 ====")

    for script, args_template in scripts2:
        target_date = date_minus1 if "Mean" in script else date_str
        args = [arg.format(date=target_date) for arg in args_template]

        full_command = ["python3", f"../ai/{script}"] + args
        print(f"실행 중: {' '.join(full_command)}")
        subprocess.run(full_command)

    current_date += timedelta(days=1)
print("Second Model 끝, 소요 시간: ", round(time.time() - now, 1), "sec")

# 후처리 부분
now = time.time()
print("후처리 시작")
for cmd in post_scripts:
    print(f"실행 중: {' '.join(cmd)}")
    subprocess.run(cmd)
print("후처리 끝, 소요 시간: ", round(time.time() - now, 1), "sec")

print("총 소요 시간: ", round(time.time() - s, 1), "sec")