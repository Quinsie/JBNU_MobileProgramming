# backend/source/utils/logger.py

import os
from datetime import datetime, time, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(BASE_DIR, "backend", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_log_date_path():
    now = datetime.now()
    if now.time() < time(5, 30):
        log_date = now - timedelta(days=1)
    else:
        log_date = now
    return os.path.join(LOG_DIR, f"run_{log_date.strftime('%Y%m%d')}.log")

def log(source, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [{source}] {message}\n"

    # 날짜별 로그 저장
    with open(get_log_date_path(), "a", encoding="utf-8") as f:
        f.write(formatted)

    # 콘솔에만 출력 (run.log는 nohup이 담당)
    print(formatted, end="", flush=True)