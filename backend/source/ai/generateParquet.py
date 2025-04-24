# backend/source/ai/generateParquet.py
# raw dynamic information을 parquet으로 정제하는 파이프라인.

import os, sys, json, time, shutil
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.getDayType import getDayType
from source.utils.convertToGrid import convert_to_grid
from source.utils.logger import log  # 로그 기록용

RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
USED_DIR = os.path.join(BASE_DIR, "data", "raw", "used")
bus_dir = os.path.join(RAW_DIR, "realtime_bus")
weather_dir = os.path.join(RAW_DIR, "weather")
traffic_dir = os.path.join(RAW_DIR, "traffic")

parse_time = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
parse_name = lambda s: datetime.strptime(s, "%Y%m%d_%H%M")
start_time = time.time()
cutoff = datetime(2025, 4, 23, 5, 30, 0)

traffic_ts_map = {
    datetime.strptime(f.replace(".json", ""), "%Y%m%d_%H%M"): f
    for f in os.listdir(traffic_dir) if f.endswith(".json")
}
weather_ts_map = {
    datetime.strptime(f.replace(".json", ""), "%Y%m%d_%H%M"): f
    for f in os.listdir(weather_dir) if f.endswith(".json")
}

weather_cache, traffic_cache = {}, {}
used_weather_files, used_traffic_files, used_bus_files = set(), set(), set()

for ts, f in weather_ts_map.items():
    try:
        with open(os.path.join(weather_dir, f)) as wf:
            weather_cache[ts] = json.load(wf)
    except:
        continue

for ts, f in traffic_ts_map.items():
    try:
        with open(os.path.join(traffic_dir, f)) as tf:
            traffic_cache[ts] = json.load(tf)
    except:
        continue

def process_bus_file(args):
    stdid, file = args
    rows = []
    route_path = os.path.join(bus_dir, stdid)
    date = parse_name(file.replace(".json", ""))
    if date < cutoff:
        return rows

    used_bus_files.add(os.path.join("realtime_bus", stdid, file))

    t0 = time.time()  # ⏱ 시작 시간 측정

    day_type = getDayType(date)
    with open(os.path.join(route_path, file), encoding="utf-8") as f:
        bus_data = json.load(f)

    start_time = datetime.strptime(f"{date.date()} {bus_data['start_time']}", "%Y-%m-%d %H:%M")
    stop_logs = bus_data.get("stop_reached_logs", [])

    for stop in stop_logs:
        ord_num = stop.get("ord")
        arrival = parse_time(stop.get("time"))
        elapsed = int((arrival - start_time).total_seconds())
        g1 = g2 = g3 = 0
        PTY = RN1 = T1H = None

        for loc in bus_data.get("location_logs", []):
            t = parse_time(loc["time"])
            if t > arrival:
                break
            mv = loc.get("matched_vertex")
            lat, lng = loc["lat"], loc["lng"]

            traffic_key = max([ts for ts in traffic_cache if ts <= t], default=None)
            if mv and traffic_key:
                node_id = mv["matched_id"]
                sub = mv["matched_sub"]
                nx_ny = f"{node_id}_{sub}"
                used_traffic_files.add(traffic_ts_map[traffic_key])
                traffic_data = traffic_cache.get(traffic_key, [])
                for record in traffic_data:
                    if record["id"] == node_id and record["sub"] == sub:
                        grade = int(record["grade"])
                        if grade == 1:
                            g1 += 1
                        elif grade == 2:
                            g2 += 1
                        elif grade == 3:
                            g3 += 1
            else:
                x, y = convert_to_grid(lat, lng)
                nx_ny = f"{x}_{y}"

            weather_key = max([ts for ts in weather_cache if ts <= t], default=None)
            if weather_key:
                used_weather_files.add(weather_ts_map[weather_key])
                weather = weather_cache.get(weather_key, {}).get(nx_ny)
                if weather:
                    PTY = int(weather["PTY"])
                    RN1 = float(weather["RN1"])
                    T1H = float(weather["T1H"])

        rows.append({
            "route_id": stdid,
            "departure_time": start_time.hour * 60 + start_time.minute,
            "day_type": {"weekday": 0, "saturday": 1, "holiday": 2}[day_type],
            "stop_order": ord_num,
            "grade_1_count": g1,
            "grade_2_count": g2,
            "grade_3_count": g3,
            "PTY": PTY,
            "RN1": RN1,
            "T1H": T1H,
            "target_elapsed_time": elapsed
        })

    log("generateParquet", f"🚏 {stdid}/{file} 처리 완료 ({len(rows)} rows, ⏱ {time.time() - t0:.2f}s)")
    return rows

def move_to_used(file_path, src_base, dst_base):
    src = os.path.join(src_base, file_path)
    dst = os.path.join(dst_base, file_path)
    log("generateParquet", f"📦 이동 시도: {src} → {dst}")
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        log("generateParquet", f"✅ 이동 완료: {file_path}")
    except Exception as e:
        log("generateParquet", f"❌ 이동 실패: {file_path} → {e}")

def generate_parquet():
    all_args = []
    for stdid in os.listdir(bus_dir):
        for file in os.listdir(os.path.join(bus_dir, stdid)):
            all_args.append((stdid, file))

    with Pool(cpu_count()) as pool:
        results = pool.map(process_bus_file, all_args)

    flat_rows = [row for group in results for row in group]
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    df = pd.DataFrame(flat_rows)
    df.to_parquet(os.path.join(PREPROCESSED_DIR, "eta_train.parquet"), index=False)

    log("generateParquet", f"✅ {len(df)} rows saved to eta_train.parquet")
    log("generateParquet", f"⏱ {time.time() - start_time:.2f} sec 소요")

    log("generateParquet", "📦 사용된 raw 파일 이동 중...")
    for f in used_bus_files:
        move_to_used(f, bus_dir, os.path.join(USED_DIR, "realtime_bus"))
    for f in used_weather_files:
        move_to_used(f, weather_dir, os.path.join(USED_DIR, "weather"))
    for f in used_traffic_files:
        move_to_used(f, traffic_dir, os.path.join(USED_DIR, "traffic"))
    log("generateParquet", "✅ 전체 파일 이동 완료.")

if __name__ == "__main__":
    generate_parquet()