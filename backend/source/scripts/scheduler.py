# backend/source/scripts/scheduler.py

import os
import sys
import json
import psutil
import subprocess
from datetime import time
from datetime import datetime
from importlib import import_module
from apscheduler.schedulers.blocking import BlockingScheduler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.getDayType import getDayType
from source.utils.logger import log  # log 함수 임포트

CACHE_DIR = os.path.join(BASE_DIR, "data", "processed", "departure_cache")

for p in psutil.process_iter(attrs=["pid", "cmdline"]): # 중복실행감지
    if p.info["pid"] != os.getpid() and "scheduler.py" in " ".join(p.info["cmdline"]):
        log("scheduler", "이미 실행 중인 scheduler.py 감지, 종료.")
        sys.exit()

def load_departure_cache(schedule_type):
    path = os.path.join(CACHE_DIR, f"{schedule_type}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_current_departures():
    now = datetime.now()
    schedule_type = getDayType(now)
    cache = load_departure_cache(schedule_type)

    hhmm = now.strftime("%H:%M")
    if hhmm in cache["cachetime"]:
        stdids = []
        for block in cache["data"]:
            if block["time"].strip() == hhmm:
                stdids.extend(block["stdid"])
        return stdids
    return []

def run_tracking_job():
    stdids = get_current_departures()
    log("scheduler", f"현재 감지된 STDIDs: {stdids}")
    now = datetime.now().strftime("%H:%M")
    t = datetime.now()
    if t.time() >= time(23, 30):
        log("scheduler", "[SYSTEM] 현재시간 23:30 :: 스케줄러 종료")
        scheduler.shutdown()
        return

    for stdid in stdids: # 병렬프로세싱으로 동시에 추적
        log("scheduler", f"{now} → {stdid} 추적 시작 (병렬 실행)")
        subprocess.Popen([
            "python3",
            "backend/source/scripts/trackSingleBus.py",
            str(stdid),
            now  # ex: "05:30"
        ])

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_tracking_job, "cron", minute="*", second=0)
    log("scheduler", "⏱실시간 버스 감시 스케줄러 시작")  # log 사용
    scheduler.start()