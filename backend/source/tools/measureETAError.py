# backend/source/tools/measureETAError.py

import os
import sys
import json
import numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# 설정
ETA_TABLE_DIR = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table")
REALTIME_RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")

def load_eta_table(target_date_str):
    eta_path = os.path.join(ETA_TABLE_DIR, f"{target_date_str}.json")
    with open(eta_path, 'r') as f:
        eta_table = json.load(f)
    return eta_table

def load_realtime_logs(target_date_str):
    logs = {}
    target_dir = os.path.join(REALTIME_RAW_DIR)
    for stdid in os.listdir(target_dir):
        stdid_dir = os.path.join(target_dir, stdid)
        if not os.path.isdir(stdid_dir):
            continue
        for file in os.listdir(stdid_dir):
            if not file.startswith(target_date_str):
                continue
            file_path = os.path.join(stdid_dir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)
                for entry in data.get('stop_reached_logs', []):
                    ord_num = str(entry['ord'])
                    time_str = entry['time'][-8:]  # HH:MM:SS
                    logs.setdefault(f"{stdid}_{file.split('_')[-1].split('.')[0]}", {})[ord_num] = time_str
    return logs

def time_to_seconds(tstr):
    h, m, s = map(int, tstr.split(':'))
    return h * 3600 + m * 60 + s

def get_time_group(hhmm):
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    minutes = hour * 60 + minute

    if 330 <= minutes < 420: return 0
    elif 420 <= minutes < 540: return 1
    elif 540 <= minutes < 720: return 2
    elif 690 <= minutes < 840: return 3
    elif 840 <= minutes < 1020: return 4
    elif 1020 <= minutes < 1140: return 5
    elif 1140 <= minutes < 1260: return 6
    else: return 7

def measure_by_time_group(eta_table, realtime_logs, group_index):
    errors = []

    for stdid_hhmm, stops in eta_table.items():
        hhmm = stdid_hhmm.split('_')[1]
        if get_time_group(hhmm) != group_index:
            continue

        realtime_stops = realtime_logs.get(stdid_hhmm, {})
        for ord_num, eta_time in stops.items():
            if ord_num not in realtime_stops:
                continue
            eta_sec = time_to_seconds(eta_time)
            real_sec = time_to_seconds(realtime_stops[ord_num])
            error = real_sec - eta_sec
            errors.append(error)

    if not errors:
        print("해당 시간대 그룹에 대한 데이터가 없습니다.")
        return

    errors = np.array(errors)
    print(f"[시간대 그룹 {group_index}] 총 비교 개수: {len(errors)}")
    print(f"평균 오차 (초): {errors.mean():.2f}")
    print(f"표준편차 (초): {errors.std():.2f}")
    print(f"최대 오차 (초): {np.max(np.abs(errors))}")

def measure_error(eta_table, realtime_logs, stdid_filter=None, hhmm_filter=None):
    errors = []

    for stdid_hhmm, stops in eta_table.items():
        if stdid_filter and not stdid_hhmm.startswith(stdid_filter):
            continue

        if hhmm_filter:
            hhmm = stdid_hhmm.split('_')[1]
            if hhmm != hhmm_filter:
                continue

        realtime_stops = realtime_logs.get(stdid_hhmm, {})
        for ord_num, eta_time in stops.items():
            if ord_num not in realtime_stops:
                continue
            eta_sec = time_to_seconds(eta_time)
            real_sec = time_to_seconds(realtime_stops[ord_num])
            error = real_sec - eta_sec
            errors.append(error)

    if not errors:
        print("비교할 데이터가 없습니다.")
        return

    errors = np.array(errors)
    print(f"총 비교 개수: {len(errors)}")
    print(f"평균 오차 (초): {errors.mean():.2f}")
    print(f"표준편차 (초): {errors.std():.2f}")
    print(f"최대 오차 (초): {np.max(np.abs(errors))}")

def main():
    from datetime import timedelta

    target_date_input = input("오차를 측정할 ETA Table 날짜를 입력하세요 (YYYYMMDD): ").strip()
    target_date = datetime.strptime(target_date_input, "%Y%m%d")
    day_after = target_date + timedelta(days=1)
    day_after_str = day_after.strftime("%Y%m%d")

    eta_table = load_eta_table(target_date_input)
    realtime_logs = load_realtime_logs(day_after_str)

    print("모드 선택:")
    print("0: 전체")
    print("1: 특정 노선")
    print("2: 특정 시간대 그룹")
    mode = input("선택: ").strip()

    if mode == "0":
        measure_error(eta_table, realtime_logs)

    elif mode == "1":
        stdid = input("노선 stdid를 입력하세요 (예: 305001088): ").strip()
        detail = input("특정 시간대만 측정할까요? 예(1)/아니오(0): ").strip()

        if detail == "1":
            hhmm = input("시간대를 입력하세요 (예: 1810): ").strip()
            measure_error(eta_table, realtime_logs, stdid_filter=stdid, hhmm_filter=hhmm)
        else:
            measure_error(eta_table, realtime_logs, stdid_filter=stdid)

    elif mode == "2":
        print("\n시간대 그룹:")
        print(" 0: 05:30~07:00 (이른 아침)")
        print(" 1: 07:00~09:00 (출근 시간)")
        print(" 2: 09:00~11:30 (아침)")
        print(" 3: 11:30~14:00 (점심)")
        print(" 4: 14:00~17:00 (오후)")
        print(" 5: 17:00~19:00 (퇴근 시간)")
        print(" 6: 19:00~21:00 (저녁)")
        print(" 7: 21:00~00:00 (밤)")
        group_index = int(input("시간대 그룹 index (0~7): ").strip())
        measure_by_time_group(eta_table, realtime_logs, group_index)

    else:
        print("잘못된 입력입니다.")

if __name__ == "__main__":
    from datetime import timedelta
    main()