import os
import re

TRAFFIC_DIR = "backend/data/raw/dynamicInfo/traffic"

# 정규 표현식: 초단위 포함된 파일 예시 YYYYmmDD_HHMMSS.json
pattern_seconds = re.compile(r"^\d{8}_\d{6}\.json$")
pattern_minutes = re.compile(r"^\d{8}_\d{4}\.json$")

seen_minute_files = set()

for file in os.listdir(TRAFFIC_DIR):
    if pattern_minutes.match(file):
        seen_minute_files.add(file.replace(".json", ""))

for file in os.listdir(TRAFFIC_DIR):
    if pattern_seconds.match(file):
        dt_full = file.replace(".json", "")  # e.g., 20250423_053012
        dt_min = dt_full[:13]  # e.g., 20250423_0530
        if dt_min in seen_minute_files:
            # 분 단위 파일이 이미 있으면 초단위 제거
            os.remove(os.path.join(TRAFFIC_DIR, file))
            print(f"Deleted redundant second-level file: {file}")