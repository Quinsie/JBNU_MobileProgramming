# backend/source/utils/logger.py

from datetime import datetime

def log(module: str, message: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{module}] {message}", flush=True)