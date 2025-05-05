# backend/source/tools/generateFirstETA.py
# 최초 1회성: 2025-04-23 raw 데이터를 기반으로 ETA 테이블 생성

import os
import sys
import json
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
SAVE_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table")
os.makedirs(SAVE_DIR, exist_ok=True)

DATE_TAG = "20250505"
eta_table = {}  # {"STDID_HHMM": {ord: arrival_time}}

def parse_arrival_time(timestr):
    try:
        return datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S").time().strftime("%H:%M:%S")
    except:
        return None

def generate():
    for stdid in os.listdir(RAW_DIR):
        stdid_dir = os.path.join(RAW_DIR, stdid)
        if not os.path.isdir(stdid_dir):
            continue

        for file in os.listdir(stdid_dir):
            if not file.startswith(DATE_TAG):
                continue

            filepath = os.path.join(stdid_dir, file)
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            stop_logs = data.get("stop_reached_logs", [])
            key = f"{stdid}_{file[9:13]}"  # STDID_HHMM
            eta_table[key] = {}

            for log in stop_logs:
                ord = str(log.get("ord"))
                time_str = parse_arrival_time(log.get("time"))
                if ord and time_str:
                    eta_table[key][ord] = time_str

    save_path = os.path.join(SAVE_DIR, f"{DATE_TAG}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(eta_table, f, ensure_ascii=False, indent=2)

    print(f"ETA 테이블 생성 완료 → {save_path} ({len(eta_table)}개 노선)")

if __name__ == "__main__":
    generate()
