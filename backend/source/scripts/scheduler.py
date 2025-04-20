# backend/source/scripts/scheduler.py

import os
import sys
import json
from datetime import datetime
from importlib import import_module
from apscheduler.schedulers.blocking import BlockingScheduler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.getDayType import getDayType

CACHE_DIR = os.path.join("backend", "data", "processed", "departure_cache")

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
    now = datetime.now().strftime("%H:%M")

    for stdid in stdids:
        print(f"{now} → {stdid} 추적 시작", flush=True)
        # 실제 추적 함수 불러오기 (각 추적 스크립트에서 track_bus 함수 사용한다고 가정)
        track_module = import_module("scripts.trackSingleBus")
        track_module.track_bus(stdid, now)

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(run_tracking_job, "cron", minute="*", second=0)
    print("⏱실시간 버스 감시 스케줄러 시작", flush=True)
    scheduler.start()