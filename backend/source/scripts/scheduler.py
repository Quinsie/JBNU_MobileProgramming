# backend/source/scripts/scheduler.py

import os
import sys
import json
import psutil
import subprocess
from datetime import datetime, time, timedelta
from importlib import import_module
from apscheduler.schedulers.blocking import BlockingScheduler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.getDayType import getDayType
from source.utils.logger import log  # log 함수 임포트

CACHE_DIR = os.path.join(BASE_DIR, "data", "processed", "departure_cache")

for p in psutil.process_iter(attrs=["pid", "cmdline"]):
    try:
        cmdline = p.info.get("cmdline") or []
        if p.info["pid"] != os.getpid() and "scheduler.py" in " ".join(cmdline):
            is_running = True
            break
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

def load_departure_cache(schedule_type):
    path = os.path.join(CACHE_DIR, f"{schedule_type}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_current_departures():
    now = datetime.now()
    future_time = now + timedelta(minutes = 1)
    schedule_type = getDayType(future_time)
    cache = load_departure_cache(schedule_type)

    hhmm = future_time.strftime("%H:%M") # 1분 전부터 탐색 대기
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
    now_time = datetime.now().time()

    if time(0, 30) < now_time < time(5, 30):
        log("scheduler", "[SYSTEM] 00시 30분 ~ 05시 30분 사이 → 스케줄러 종료")
        scheduler.shutdown(wait=False)
        return

    for stdid in stdids: # 병렬프로세싱으로 동시에 추적
        hhmm = (datetime.now() + timedelta(minutes=1)).strftime("%H:%M")
        log("scheduler", f"{now} → {stdid} 추적 시작 (출발 예정: {hhmm})")
        subprocess.Popen([
            "python3",
            "source/scripts/trackSingleBus.py",
            str(stdid),
            hhmm  # ex: "05:30"
        ], cwd = BASE_DIR)

def clean_zombie_processes():
    cleaned = 0
    for p in psutil.process_iter(attrs=["pid", "status", "ppid"]):
        try:
            if p.info["status"] == psutil.STATUS_ZOMBIE:
                os.waitpid(p.info["pid"], os.WNOHANG)  # 자식 프로세스 수거
                cleaned += 1
        except (ChildProcessError, PermissionError, ProcessLookupError):
            continue

    if cleaned > 0:
        log("scheduler", f"[CLEANER] 좀비 프로세스 {cleaned}개 수거 완료")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_tracking_job, "cron", minute="*", second=0)
    scheduler.add_job(clean_zombie_processes, "cron", minute="*", second=30)
    log("scheduler", "실시간 버스 감시 스케줄러 시작")
    scheduler.start()