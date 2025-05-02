import os
import sys
import json
from datetime import datetime, timedelta

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

def get_yesterday_file_list():
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    files = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith(yesterday) and f.endswith(".json")
    ]
    entries = []
    for f in files:
        try:
            date_part, time_part = f.replace(".json", "").split("_")
            dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M")
            with open(os.path.join(DATA_DIR, f), "r", encoding="utf-8") as jf:
                data = json.load(jf)
            entries.append({
                "filename": f,
                "datetime": dt,
                "path": os.path.join(DATA_DIR, f),
                "data": data
            })
        except Exception as e:
            log("fillNullWeather", f"❌ 파일 무시: {f} → {e}")
    return sorted(entries, key=lambda x: x["datetime"])

def fill_yesterday_missing():
    entries = get_yesterday_file_list()
    if not entries:
        log("fillNullWeather", "어제 날짜 파일 없음. 종료.")
        return

    date = entries[0]["datetime"].strftime("%Y%m%d")
    log("fillNullWeather", f"어제 날짜 {date} 총 {len(entries)}개 파일 처리")

    for i, entry in enumerate(entries):
        current_data = entry["data"]
        updated = False

        for key, val in current_data.items():
            if is_null_weather(val):
                for past_entry in reversed(entries[:i]):
                    past_val = past_entry["data"].get(key)
                    if not is_null_weather(past_val):
                        log("fillNullWeather", f"{entry['filename']}의 {key} ← {past_entry['filename']}")
                        current_data[key] = past_val
                        updated = True
                        break

        if updated:
            with open(entry["path"], "w", encoding="utf-8") as f:
                json.dump(current_data, f, ensure_ascii=False, indent=2)
            log("fillNullWeather", f"{entry['filename']} 저장 완료")

if __name__ == "__main__":
    fill_yesterday_missing()