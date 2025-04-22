# backend/source/scripts/runAll.py

import os
import sys
import psutil
import asyncio
import traceback
import subprocess
from datetime import datetime, time

# 상대 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); sys.path.append(BASE_DIR)
from utils.logger import log  # logger 모듈에서 log 함수 가져옴

for p in psutil.process_iter(attrs=["pid", "cmdline"]):
    if p.info["pid"] != os.getpid() and "runAll.py" in " ".join(p.info["cmdline"]):
        log("runAll", "이미 실행 중인 runAll.py 감지, 종료.")
        sys.exit()

def is_within_active_hours():
    now = datetime.now().time()
    return now >= time(5, 30) or now <= time(0, 30)

async def run_weather():
    while True:
        if is_within_active_hours():
            log("runAll", "날씨 수집 시작")
            subprocess.Popen(["python3", "weatherCollecter.py"], cwd="backend/source/scripts")
            await asyncio.sleep(1800)  # 활성 시간: 30분 간격
        else:
            log("runAll", "날씨 수집 대기 중 (비활성 시간)")
            await asyncio.sleep(900)  # 비활성 시간: 15분 간격 로그 출력

async def run_traffic():
    log("runAll", "[DEBUG] run_traffic 함수 진입")
    while True:
        if is_within_active_hours():
            log("runAll", "교통 수집 시작")
            subprocess.Popen(["python3", "trafficCollector.py"], cwd="backend/source/scripts")
            await asyncio.sleep(60) # 기존 10초 -> 1분
        else:
            log("runAll", "교통 수집 대기 중 (비활성 시간)")
            await asyncio.sleep(900)  # 15분 간격 로그

async def run_scheduler():
    has_started = False
    while True:
        if is_within_active_hours():
            if not has_started:
                is_running = False
                for p in psutil.process_iter(attrs=["pid", "cmdline"]):
                    try:
                        cmdline = p.info.get("cmdline") or []
                        if p.info["pid"] != os.getpid() and "scheduler.py" in " ".join(cmdline):
                            is_running = True
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                if is_running:
                    log("runAll", "이미 scheduler.py 실행 중, 생략")
                else:
                    log("runAll", "스케줄러 실행")
                    subprocess.Popen(["python3", "scheduler.py"], cwd="backend/source/scripts")
                    has_started = True
            await asyncio.sleep(60)
        else:
            log("runAll", "스케줄러 대기 중 (비활성 시간)")
            has_started = False
            await asyncio.sleep(900)  # 15분 간격 로그

async def main():
    now = datetime.now()
    log("runAll", f"[SYSTEM] {now} 실행 시작됨")
    await asyncio.gather(
        run_weather(),
        run_traffic(),
        run_scheduler()
    )

if __name__ == "__main__":
    asyncio.run(main())