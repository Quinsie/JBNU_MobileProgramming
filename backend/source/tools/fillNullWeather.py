# backend/source/tools/fillNullWeather.py

import os
import json
from collections import defaultdict
from datetime import datetime

DATA_DIR = "data/raw/dynamicInfo/weather"
LOG_PATH = "source/tools/fillNullWeather_all.log"

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
    log_lines = []

    for date, entries in data_by_date.items():
        print(f"\n📅 처리 중: {date} ({len(entries)}개 파일)")

        for i, entry in enumerate(entries):
            current_data = entry["data"]
            updated = False

            for key, val in current_data.items():
                if is_null_weather(val):
                    for past_entry in reversed(entries[:i]):
                        past_val = past_entry["data"].get(key)
                        if not is_null_weather(past_val):
                            log_line = f"[{date}] {entry['filename']}의 {key} ← {past_entry['filename']}"
                            print(f"  ⮕ {log_line}")
                            log_lines.append(log_line)
                            current_data[key] = past_val
                            updated = True
                            break

            if updated:
                with open(entry["path"], "w", encoding="utf-8") as f:
                    json.dump(current_data, f, ensure_ascii=False, indent=2)
                print(f"✅ {entry['filename']} 저장 완료")

    # 로그 저장
    if log_lines:
        with open(LOG_PATH, "w", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_lines))
        print(f"\n📝 로그 저장됨 → {LOG_PATH}")

if __name__ == "__main__":
    all_data = load_all_weather_files()
    fill_missing_data(all_data)