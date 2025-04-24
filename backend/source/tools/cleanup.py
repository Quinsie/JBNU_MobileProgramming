# backend/source/tools/cleanup.py
# 2025-04-23 05:30 이전 데이터 삭제하는 코드

import os
from datetime import datetime

CUTOFF = datetime(2025, 4, 23, 5, 30)

BASE_DIR = "backend/data/raw/dynamicInfo"
folders = ["realtime_bus", "traffic", "weather"]

def extract_datetime(filename):
    try:
        name = filename.replace(".json", "")
        name = name.split("_")[0] + "_" + name.split("_")[1][:4]  # YYYYmmDD_HHMM
        return datetime.strptime(name, "%Y%m%d_%H%M")
    except:
        return None

for folder in folders:
    folder_path = os.path.join(BASE_DIR, folder)
    if folder == "realtime_bus":
        for stdid in os.listdir(folder_path):
            stdid_path = os.path.join(folder_path, stdid)
            for file in os.listdir(stdid_path):
                dt = extract_datetime(file)
                if dt and dt < CUTOFF:
                    os.remove(os.path.join(stdid_path, file))
                    print(f"Deleted {folder}/{stdid}/{file}")
    else:
        for file in os.listdir(folder_path):
            dt = extract_datetime(file)
            if dt and dt < CUTOFF:
                os.remove(os.path.join(folder_path, file))
                print(f"Deleted {folder}/{file}")