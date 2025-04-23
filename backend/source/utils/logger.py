# backend/source/utils/logger.py

import os
from datetime import datetime, time, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_current_log_file = None

def is_active_hours():
    now = datetime.now().time()
    return now >= time(5, 30) or now <= time(0, 30)

def get_log_path():
    now = datetime.now()
    if now.time() < time(5, 30):
        log_date = now - timedelta(days=1)
    else:
        log_date = now
    return os.path.join(LOG_DIR, f"run_{log_date.strftime('%Y%m%d')}.log")

def log(source, message):
    global _current_log_file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [{source}] {message}"

    # 콘솔 출력 (nohup 출력 리다이렉트로 run.log에 자동 저장됨)
    print(formatted, flush=True)

    # 기록용 로그 (활성 시간일 때만)
    if is_active_hours():
        log_path = get_log_path()
        if _current_log_file != log_path:
            _current_log_file = log_path
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n 로그 시작: {timestamp}\n{'='*50}\n")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")