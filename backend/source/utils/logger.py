# backend/source/utils/logger.py

from datetime import datetime
import os

LOG_DIR = "backend/logs"

def log(module: str, message: str):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y%m%d")

    formatted = f"[{timestamp}] [{module}] {message}"

    # 경로 생성
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"run_{date_str}.log")

    # 출력
    print(formatted, flush=True)

    # 파일에도 기록
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")