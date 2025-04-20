# backend/source/scripts/runAll.py
# nohup python3 backend/source/scripts/run_everything.py >> backend/logs/run.log 2>&1 &

import os
import sys
import asyncio
import traceback
from datetime import datetime, time

# 상대 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__)))

def is_within_active_hours():
    now = datetime.now().time()
    return time(5, 30) <= now <= time(23, 30)

async def run_weather():
    while True:
        if is_within_active_hours():
            try:
                print("🌤 날씨 수집 시작")
                os.system("python weatherCollector.py")
            except Exception:
                traceback.print_exc()
        else:
            print("⏳ 날씨 수집 대기 중 (비활성 시간)")
        await asyncio.sleep(1800)  # 30분 간격

async def run_traffic():
    while True:
        if is_within_active_hours():
            try:
                print("🚦 교통 수집 시작")
                os.system("python trafficCollector.py")
            except Exception:
                traceback.print_exc()
        else:
            print("⏳ 교통 수집 대기 중 (비활성 시간)")
        await asyncio.sleep(10)

async def run_scheduler():
    while True:
        if is_within_active_hours():
            try:
                print("🚌 스케줄러 실행")
                os.system("python scheduler.py")
            except Exception:
                traceback.print_exc()
        else:
            print("⏳ 스케줄러 대기 중 (비활성 시간)")
        await asyncio.sleep(60)

async def main():
    now = datetime.now()
    print("[SYSTEM]", now, "실행 시작됨")
    await asyncio.gather(
        run_weather(),
        run_traffic(),
        run_scheduler()
    )

if __name__ == "__main__":
    asyncio.run(main())