# backend/source/helpers/autoTrainer.py

import time
import subprocess
from datetime import datetime, timedelta

s = time.time()
# 시작일자, 종료일자 설정
start_date = datetime.strptime("20250520", "%Y%m%d")
end_date = datetime.strptime("20250601", "%Y%m%d")  # 종료일 포함

# loop 이전 사전 스크립트
pre_scripts = [
    #["python3", "../ai/buildFirstReplayParquetStart.py", "--date", "20250519"],
    ["python3", "../ai/trainFirstETA.py", "--date", "20250519", "--mode", "replay"],
    ["python3", "../ai/generateFirstETA.py", "--date", "20250519"],
]

# 실행할 스크립트 및 인자 정의
scripts = [
    ("buildFirstReviewParquet.py", ["--date", "{date}"]),
    ("trainFirstETA.py", ["--date", "{date}", "--mode", "self_review"]),
    ("buildFirstReplayParquet.py", ["--date", "{date}"]),
    ("trainFirstETA.py", ["--date", "{date}", "--mode", "replay"]),
    ("generateMeanElapsed.py", ["--date", "{date}", "--mode", "append"]),
    ("generateMeanInterval.py", ["--date", "{date}", "--mode", "append"]),
    ("generateFirstETA.py", ["--date", "{date}"]),
]

now = time.time()
print("전처리 시작")
for cmd in pre_scripts:
    print(f"실행 중: {' '.join(cmd)}")
    subprocess.run(cmd)
print("전처리 끝, 소요 시간: ", round(time.time() - now, 1), "sec")

now = time.time()
print("ai/ 시작")
# 날짜별 루프
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")
    date_minus1 = (current_date - timedelta(days=1)).strftime("%Y%m%d")
    print(f"\n==== [{date_str}] 시작 ====")
    
    for script, args_template in scripts:
        # 날짜 결정 (일부 스크립트는 하루 전 날짜 사용)
        target_date = date_minus1 if "Mean" in script else date_str
        args = [arg.format(date=target_date) for arg in args_template]
        full_command = ["python3", f"../ai/{script}"] + args
        print(f"실행 중: {' '.join(full_command)}")
        subprocess.run(full_command)
    
    current_date += timedelta(days=1)
print("ai/ 끝, 소요 시간: ", round(time.time() - now, 1), "sec")

# now = time.time()
# print("ai2/ 시작")
# current_date = start_date
# while current_date <= end_date:
#     date_str = current_date.strftime("%Y%m%d")
#     print(f"\n==== [{date_str}] 시작 ====")
    
#     for script, args_template in scripts:
#         # 날짜 결정 (일부 스크립트는 하루 전 날짜 사용)
#         if "Mean" in script: continue
#         target_date =  date_str
#         args = [arg.format(date=target_date) for arg in args_template]
#         full_command = ["python3", f"../ai2/{script}"] + args
#         print(f"실행 중: {' '.join(full_command)}")
#         subprocess.run(full_command)
    
#     current_date += timedelta(days=1)
# print("ai2/ 끝, 소요 시간: ", time.time() - now, "sec")

print("총 소요 시간: ", round(time.time() - s, 1), "sec")