import os
import json
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ETA_TABLE_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", "first_model")
REALTIME_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")

def time_to_seconds(tstr):
    h, m, s = map(int, tstr.split(":"))
    return h * 3600 + m * 60 + s

def load_eta_table(date_str):
    path = os.path.join(ETA_TABLE_DIR, f"{date_str}.json")
    if not os.path.exists(path):
        print("ETA 테이블이 존재하지 않습니다:", path)
        exit()
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_realtime_logs(date_str):
    logs = {}
    target_dir = os.path.join(REALTIME_RAW_DIR)
    for stdid in os.listdir(target_dir):
        stdid_dir = os.path.join(target_dir, stdid)
        if not os.path.isdir(stdid_dir): continue
        for fname in os.listdir(stdid_dir):
            if not fname.startswith(date_str): continue
            fpath = os.path.join(stdid_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                hhmm = fname.split("_")[-1].split(".")[0]
                key = f"{stdid}_{hhmm}"
                for entry in data.get("stop_reached_logs", []):
                    ord_str = str(entry["ord"])
                    time_str = entry["time"][-8:]  # HH:MM:SS
                    logs.setdefault(key, {})[ord_str] = time_str
            except Exception as e:
                continue
    return logs

def analyze_distribution(eta_table, realtime_logs):
    errors = []

    for stdid_hhmm, eta_dict in eta_table.items():
        realtime_dict = realtime_logs.get(stdid_hhmm, {})
        for ord_key, eta_time in eta_dict.items():
            if ord_key not in realtime_dict:
                continue
            try:
                eta_sec = time_to_seconds(eta_time)
                real_sec = time_to_seconds(realtime_dict[ord_key])
                diff = real_sec - eta_sec
                errors.append(diff)
            except:
                continue

    if not errors:
        print("비교할 데이터가 없습니다.")
        return

    err_array = np.array(errors)
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

def main():
    date_input = input("ETA Table 날짜를 입력하세요 (YYYYMMDD): ").strip()
    eta_table = load_eta_table(date_input)

    next_date = datetime.strptime(date_input, "%Y%m%d") + timedelta(days=1)
    realtime_date_str = next_date.strftime("%Y%m%d")
    realtime_logs = load_realtime_logs(realtime_date_str)

    analyze_distribution(eta_table, realtime_logs)

if __name__ == "__main__":
    main()