# backend/source/tools/cleanupTraffic.py
# 초단위 수집했던 파일 삭제

import os
import re

TRAFFIC_DIR = "backend/data/raw/dynamicInfo/traffic"

# 초단위: 무조건 지우도록 수정
pattern_seconds = re.compile(r"^\d{8}_\d{6}\.json$")

deleted = 0
for file in os.listdir(TRAFFIC_DIR):
    if pattern_seconds.match(file):
        os.remove(os.path.join(TRAFFIC_DIR, file))
        print(f"🧹 Deleted: {file}")
        deleted += 1

if deleted == 0:
    print("No second-level traffic files found. All clean!")
else:
    print(f"{deleted} second-level traffic files deleted.")