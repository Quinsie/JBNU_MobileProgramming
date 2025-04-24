# backend/source/tools/departureCacheGenerator.py

import os
import json
from collections import defaultdict

# 경로 설정
input_dir = os.path.join("backend", "data", "raw", "staticInfo", "departure_timetables")
output_dir = os.path.join("backend", "data", "processed")
os.makedirs(output_dir, exist_ok=True)

# 요일별 캐시 구조
cache = {
    "weekday": defaultdict(list),
    "saturday": defaultdict(list),
    "holiday": defaultdict(list),
}

# 파일 반복
for fname in os.listdir(input_dir):
    if not fname.endswith(".json"):
        continue

    stdid = fname.replace(".json", "")
    fpath = os.path.join(input_dir, fname)

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for day_type in ["weekday", "saturday", "holiday"]:
        times = data.get(day_type, [])
        if not times:
            continue  # 시간표 없음은 저장하지 않음

        for t in times:
            clean_time = t.strip()
            cache[day_type][clean_time].append(int(stdid))

# 요일별 캐시 저장
for day_type in ["weekday", "saturday", "holiday"]:
    result = {
        "cachetime": sorted(cache[day_type].keys()),
        "data": [],
    }

    for t in sorted(cache[day_type].keys()):
        result["data"].append({
            "time": t,
            "stdid": sorted(cache[day_type][t]),
        })

    save_path = os.path.join(output_dir, f"{day_type}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {save_path}")