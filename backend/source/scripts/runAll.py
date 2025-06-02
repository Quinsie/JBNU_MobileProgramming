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

async def run_zombie_cleaner():
    while True:
        cleaned = 0
        for p in psutil.process_iter(attrs=["pid", "status"]):
            try:
                if p.info["status"] == psutil.STATUS_ZOMBIE:
                    os.waitpid(p.info["pid"], os.WNOHANG)
                    cleaned += 1
            except (ChildProcessError, PermissionError, ProcessLookupError):
                continue
        if cleaned > 0:
            log("runAll", f"[CLEANER] 좀비 프로세스 {cleaned}개 수거 완료")
        await asyncio.sleep(60)  # 1분마다 반복

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
            await asyncio.sleep(60)  # 수집 여유 시간
            # log("runAll", "교통 혼잡도 매핑 시작")
            # subprocess.Popen(["python3", "mapCongestion.py"], cwd="backend/source/scripts")
            # await asyncio.sleep(55)  # 남은 시간 대기 (총 60초 주기)
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

async def run_relay_second_model():
    has_started = False
    while True:
        if is_within_active_hours():
            if not has_started:
                is_running = False
                for p in psutil.process_iter(attrs=["pid", "cmdline"]):
                    try:
                        cmdline = p.info.get("cmdline") or []
                        if p.info["pid"] != os.getpid() and "relaySecondModel.py" in " ".join(cmdline):
                            is_running = True
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                if is_running:
                    log("runAll", "이미 relaySecondModel.py 실행 중, 생략")
                else:
                    log("runAll", "relaySecondModel 실행")
                    subprocess.Popen(["python3", "relaySecondModel.py"], cwd="backend/source/ai")
                    has_started = True
            await asyncio.sleep(60)
        else:
            log("runAll", "relaySecondModel 대기 중 (비활성 시간)")
            has_started = False
            await asyncio.sleep(900)

async def run_forecast():
    has_run_today = False
    while True:
        now = datetime.now()
        current_time = now.time()

        if time(0, 0) <= current_time <= time(0, 30):
            if not has_run_today:
                log("runAll", "단기예보 수집 시작")
                subprocess.Popen(["python3", "forecastCollector.py"], cwd="backend/source/scripts")
                has_run_today = True
            else:
                log("runAll", "단기예보 이미 수집 완료, 대기 중")
        else:
            if current_time >= time(1, 0):
                has_run_today = False  # 새벽 1시 넘으면 다음날 준비
            log("runAll", "단기예보 수집 대기 중 (비활성 시간)")

        await asyncio.sleep(1800)  # 30분마다 체크

async def main():
    now = datetime.now()
    log("runAll", f"[SYSTEM] {now} 실행 시작됨")
    await asyncio.gather(
        run_weather(),
        run_traffic(),
        run_scheduler(),
        run_forecast(),
        run_relay_second_model(),
        run_zombie_cleaner()
    )

if __name__ == "__main__":
    asyncio.run(main())