# backend/source/scripts/trackSingleBus.py

import os
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from haversine import haversine_distance

STOP_DIR = Path("backend/data/raw/staticInfo/stops")
VTX_MAP_DIR = Path("backend/data/processed/vtx_mapped")
SAVE_DIR = Path("backend/data/raw/dynamicInfo/realtime_bus")

URL = "http://www.jeonjuits.go.kr/bis/selectBisRouteLocationList.do"
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
}

def load_stop_list(stdid):
    path = STOP_DIR / f"{stdid}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["resultList"]

def load_vtx_map(stdid):
    path = VTX_MAP_DIR / f"{stdid}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["resultList"]

def get_matched_vertex(lat, lng, vtx_list):
    closest = None
    min_dist = float("inf")
    for vtx in vtx_list:
        dist = haversine_distance(lat, lng, vtx["LAT"], vtx["LNG"])
        if dist < min_dist:
            min_dist = dist
            closest = {
                "matched_id": vtx["MATCHED_ID"],
                "matched_sub": vtx["MATCHED_SUB"],
                "distance": dist
            }
    return closest if min_dist <= 0.1 else None

def track_bus(stdid, start_time_str):
    print(f"{stdid} 버스 {start_time_str} 출발분 추적 시작")

    stop_list = load_stop_list(stdid)
    vtx_list = load_vtx_map(stdid)
    end_ord = max(s["STOP_ORD"] for s in stop_list)
    end_ord_minus1 = end_ord - 1

    start_time = datetime.strptime(start_time_str, "%H:%M").time()
    date_str = datetime.now().strftime("%Y%m%d")
    bus_file_dir = SAVE_DIR / str(stdid)
    bus_file_dir.mkdir(parents=True, exist_ok=True)
    bus_file_path = bus_file_dir / f"{date_str}_{start_time_str.replace(':', '')}.json"

    tracked_plate = None
    reached_ords = set()
    location_logs = []
    stop_reached_logs = []

    last_movement = time.time()
    reached_end_minus1 = False
    end_check_start = None

    try:
        while True:
            now = datetime.now().time()
            if now < start_time:
                time.sleep(10)
                continue

            try:
                res = requests.post(URL, headers=HEADERS, data={"locale": "ko-kr", "routeId": stdid})
                if not res.text.strip():
                    raise ValueError("빈 응답")
                data = res.json()
                bus_list = data.get("busPosList", [])
            except Exception as e:
                print(f"[에러] API 실패: {e}")
                time.sleep(5)
                continue

            target_bus = None
            for bus in bus_list:
                if tracked_plate:
                    if bus["PLATE_NO"].strip() == tracked_plate:
                        target_bus = bus
                        break
                else:
                    if bus["CURRENT_NODE_ORD"] in [1, 2]:
                        tracked_plate = bus["PLATE_NO"].strip()
                        target_bus = bus
                        print(f"➡️ 추적 시작: {tracked_plate} (ORD {bus['CURRENT_NODE_ORD']})")
                        if bus["CURRENT_NODE_ORD"] == 2:
                            reached_ords.add(1)
                            stop_reached_logs.append({
                                "ord": 1,
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "note": "ORD 2에서 시작했기 때문에 ORD 1 강제 삽입"
                            })
                            print(f"🟡 ORD 1 강제 삽입")
                        break

            if not target_bus:
                print("[대기] 대상 버스 없음")
                if reached_end_minus1 and end_check_start and time.time() - end_check_start > 60:
                    print(f"종점 도달(ORD {end_ord} 감지 실패, ORD {end_ord_minus1} 이후 사라짐)")
                    break
                if time.time() - last_movement > 15 * 60:
                    print(f"타임아웃: 15분 이상 정체")
                    break
                time.sleep(5)
                continue

            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ord = target_bus["CURRENT_NODE_ORD"]
            lat = target_bus["LAT"]
            lng = target_bus["LNG"]
            matched = get_matched_vertex(lat, lng, vtx_list)

            if ord not in reached_ords:
                reached_ords.add(ord)
                stop_reached_logs.append({
                    "ord": ord,
                    "time": now_time
                })
                last_movement = time.time()
                print(f"ORD {ord} 도착: {now_time}")

                if ord == end_ord_minus1:
                    reached_end_minus1 = True
                    end_check_start = time.time()

            location_logs.append({
                "time": now_time,
                "lat": lat,
                "lng": lng,
                "matched_vertex": matched
            })

            if ord == end_ord:
                print(f"종점 도달")
                break

            time.sleep(10)

    except KeyboardInterrupt:
        print("수동 중단됨. 로그 저장 중...")

    # 종료 또는 중단 시 저장
    with open(bus_file_path, "w", encoding="utf-8") as f:
        json.dump({
            "plate_no": tracked_plate,
            "start_time": start_time_str,
            "location_logs": location_logs,
            "stop_reached_logs": stop_reached_logs
        }, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {bus_file_path}")

if __name__ == "__main__":
    stdid = 305001892
    start_time = "22:55"
    track_bus(stdid, start_time)