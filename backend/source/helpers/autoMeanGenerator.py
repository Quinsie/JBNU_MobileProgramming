# backend/source/helpers/autoMeanGenerator.py

import time
import subprocess
from datetime import datetime, timedelta

# 시작일자, 종료일자 설정
start_date = datetime.strptime("20250506", "%Y%m%d")
end_date = datetime.strptime("20250527", "%Y%m%d")  # 종료일 포함

# loop 이전 사전 스크립트
pre_scripts = [
    ["python3", "../ai/generateMeanNode.py", "--date", "20250505", "--mode", "init"]
]

# 실행할 스크립트 및 인자 정의
scripts = [
    ("../ai/generateMeanNode.py", ["--date", "{date}", "--mode", "append"])
]

now = time.time()
for cmd in pre_scripts:
    print(f"실행 중: {' '.join(cmd)}")
    subprocess.run(cmd)

# 날짜별 루프
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y%m%d")
    
    for script, args_template in scripts:
        args = [arg.format(date=date_str) for arg in args_template]
        full_command = ["python3", f"{script}"] + args
        print(f"실행 중: {' '.join(full_command)}")
        subprocess.run(full_command)
    
    current_date += timedelta(days=1)

print("소요 시간: ", round(time.time() - now, 1), "sec")