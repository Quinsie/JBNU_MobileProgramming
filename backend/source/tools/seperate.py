# backend/source/tools/seperate.py
# 기존 버스 로그 트래킹 파일을 분리 (전처리 속도 향상을 위한 작업)

import os
import json

RAW_BASE = "backend/data/raw/dynamicInfo/realtime_bus"
POS_BASE = "backend/data/raw/dynamicInfo/realtime_pos"

os.makedirs(POS_BASE, exist_ok=True)

count = 0

for stdid in os.listdir(RAW_BASE):
    bus_path = os.path.join(RAW_BASE, stdid)
    if not os.path.isdir(bus_path):
        continue

    pos_path = os.path.join(POS_BASE, stdid)
    os.makedirs(pos_path, exist_ok=True)

    for file in os.listdir(bus_path):
        file_path = os.path.join(bus_path, file)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # location_logs 분리
        if "location_logs" in data:
            loc_data = {"location_logs": data["location_logs"]}
            del data["location_logs"]

            with open(os.path.join(pos_path, file), "w", encoding="utf-8") as pf:
                json.dump(loc_data, pf, ensure_ascii=False, indent=2)

        # plate_no, start_time 제거
        data.pop("plate_no", None)
        data.pop("start_time", None)

        with open(file_path, "w", encoding="utf-8") as bf:
            json.dump(data, bf, ensure_ascii=False, indent=2)

        count += 1
        print(f"정리 완료: {stdid}/{file}")

print(f"\n총 {count}개 파일 정리 완료.")