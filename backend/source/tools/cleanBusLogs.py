# backend/source/tools/cleanBusLogs.py

import os
import json
import time
import multiprocessing
from datetime import datetime
from glob import glob

# ===== 경로 설정 =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BUS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
POS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_pos")

# 파일명에서 기준 날짜 추출 (YYYY-mm-DD 형식)
def extract_file_date(filename: str) -> str:
    try:
        yyyymmdd = os.path.basename(filename).split("_")[0]
        dt = datetime.strptime(yyyymmdd, "%Y%m%d")
        return dt.strftime("%Y-%m-%d")
    except:
        return None

# 하나의 realtime_bus와 대응되는 pos 로그를 정제
# return: (파일명, 삭제된 ORD 수, 삭제된 10초 로그 수)
def clean_pair(file_path: str) -> tuple:
    try:
        stdid = os.path.basename(os.path.dirname(file_path))
        filename = os.path.basename(file_path)
        file_date_str = extract_file_date(filename)
        if file_date_str is None:
            return (filename, 0, 0)

        # ===== 1. bus 로그 읽기 =====
        with open(file_path, encoding="utf-8") as f:
            bus_data = json.load(f)

        logs = bus_data.get("stop_reached_logs", [])
        cutoff_time = None

        for i in range(1, len(logs)):
            t1 = datetime.strptime(logs[i - 1]["time"], "%Y-%m-%d %H:%M:%S")
            t2 = datetime.strptime(logs[i]["time"], "%Y-%m-%d %H:%M:%S")

            # [조건 1] 10분 이상 차이
            if (t2 - t1).total_seconds() > 600:
                cutoff_time = t2
                break

            # [조건 2] 날짜가 다르면 컷오프
            if logs[i]["time"].split()[0] != file_date_str:
                cutoff_time = t2
                break

        if cutoff_time:
            logs = [log for log in logs if datetime.strptime(log["time"], "%Y-%m-%d %H:%M:%S") < cutoff_time]

        ord_deleted = len(bus_data.get("stop_reached_logs", [])) - len(logs)
        bus_data["stop_reached_logs"] = logs

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(bus_data, f, ensure_ascii=False, indent=2)

        # ===== 2. pos 로그 처리 =====
        pos_file = os.path.join(POS_DIR, stdid, filename)
        pos_deleted = 0
        if os.path.exists(pos_file):
            with open(pos_file, encoding="utf-8") as f:
                pos_data = json.load(f)

            if cutoff_time:
                pos_data = [row for row in pos_data if datetime.strptime(row["time"], "%Y-%m-%d %H:%M:%S") < cutoff_time]
                pos_deleted = len(pos_data)

            with open(pos_file, "w", encoding="utf-8") as f:
                json.dump(pos_data, f, ensure_ascii=False, indent=2)

        return (filename, ord_deleted, pos_deleted)

    except Exception as e:
        return (file_path, f"[ERROR] {e}", 0)

def main():
    now = time.time()
    all_files = glob(os.path.join(BUS_DIR, "*", "*.json"))
    print(f"[INFO] 전체 파일 개수: {len(all_files)}")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(clean_pair, all_files)

    print("\n=== 삭제 통계 ===")
    for filename, ord_del, pos_del in results:
        print(f"{filename} | ORD 삭제: {ord_del} | 10초 로그 삭제: {pos_del}")

    print(f"\n[완료] 총 처리 파일 수: {len(results)}")
    print("소요 시간: ", time.time() - now, "sec")

if __name__ == "__main__":
    main()