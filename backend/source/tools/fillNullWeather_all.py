# backend/source/tools/fillNullWeather_all.py

import os
import sys
import json
from collections import defaultdict
from datetime import datetime

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")
from source.utils.logger import log

def is_null_weather(val):
    return (
        val is None
        or not isinstance(val, dict)
        or all(v is None for v in val.values())
    )

def load_all_weather_files():
    data_by_date = defaultdict(list)

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            date_part, time_part = fname.replace(".json", "").split("_")
            full_time = datetime.strptime(date_part + time_part, "%Y%m%d%H%M")
        except ValueError:
            continue

        path = os.path.join(DATA_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data_by_date[date_part].append({
            "filename": fname,
            "datetime": full_time,
            "path": path,
            "data": data
        })

    for date in data_by_date:
        data_by_date[date].sort(key=lambda x: x["datetime"])

    return data_by_date

def fill_missing_data(data_by_date):
    for date, entries in data_by_date.items():
        log("fillNullWeather_all", f"\n📅 처리 중: {date} ({len(entries)}개 파일)")

        for i, entry in enumerate(entries):
            current_data = entry["data"]
            updated = False

            for key, val in current_data.items():
                if is_null_weather(val):
                    for past_entry in reversed(entries[:i]):
                        past_val = past_entry["data"].get(key)
                        if not is_null_weather(past_val):
                            log("fillNullWeather_all", f"{entry['filename']}의 {key} ← {past_entry['filename']}")
                            current_data[key] = past_val
                            updated = True
                            break

            if updated:
                with open(entry["path"], "w", encoding="utf-8") as f:
                    json.dump(current_data, f, ensure_ascii=False, indent=2)
                log("fillNullWeather_all", f"✅ {entry['filename']} 저장 완료")

if __name__ == "__main__":
    all_data = load_all_weather_files()
    fill_missing_data(all_data)