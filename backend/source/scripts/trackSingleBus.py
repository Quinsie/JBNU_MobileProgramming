# backend/source/scripts/trackSingleBus.py

import os
import sys
import json
import time
import requests
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.haversine import haversine_distance
from source.utils.logger import log

# 경로 설정
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")
PAIR_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_pair")
BUS_SAVE_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
POS_SAVE_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_pos")

URL = "http://www.jeonjuits.go.kr/bis/selectBisRouteLocationList.do"
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "User-Agent": "Mozilla/5.0",
    "X-Requested-With": "XMLHttpRequest",
}

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_closest_route_node(lat, lng, ord, node_list, ord_pair_map, prev_node_id):
    node_ids = ord_pair_map.get(str(ord))
    if not node_ids:
        return None

    candidates = [n for n in node_list if n["NODE_ID"] in node_ids and (prev_node_id is None or n["NODE_ID"] >= prev_node_id)]

    closest = None
    min_dist = float("inf")

    for node in candidates:
        dist = haversine_distance(lat, lng, node["LAT"], node["LNG"])
        if dist < min_dist:
            min_dist = dist
            closest = node

    if closest and min_dist <= 200:
        closest = closest.copy()  # 원본 보호
        closest["distance"] = min_dist
        return closest

    return None

def track_bus(stdid, start_time_str):
    log("trackSingleBus", f"{stdid} 버스 {start_time_str} 출발분 추적 시작")

    stop_list = load_json(os.path.join(STOP_DIR, f"{stdid}.json"))["resultList"]
    node_list = load_json(os.path.join(NODE_DIR, f"{stdid}.json"))
    ord_pair_map = load_json(os.path.join(PAIR_DIR, f"{stdid}.json"))

    end_ord = max(s["STOP_ORD"] for s in stop_list)
    end_ord_minus1 = end_ord - 1

    start_time = datetime.strptime(start_time_str, "%H:%M").time()
    date_str = datetime.now().strftime("%Y%m%d")
    bus_file_dir = os.path.join(BUS_SAVE_DIR, str(stdid))
    pos_file_dir = os.path.join(POS_SAVE_DIR, str(stdid))
    os.makedirs(bus_file_dir, exist_ok=True)
    os.makedirs(pos_file_dir, exist_ok=True)
    filename = f"{date_str}_{start_time_str.replace(':', '')}.json"
    bus_file_path = os.path.join(bus_file_dir, filename)
    pos_file_path = os.path.join(pos_file_dir, filename)

    tracked_plate = None
    reached_ords = set()
    location_logs = []
    stop_reached_logs = []

    last_movement = time.time()
    reached_end_minus1 = False

    current_ord = None
    ord_stay_start = None
    last_node_id = None

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
                log("trackSingleBus", f"[에러] API 실패: {e}")
                time.sleep(5)
                continue

            target_bus = None
            for bus in bus_list:
                if tracked_plate:
                    if bus["PLATE_NO"].strip() == tracked_plate:
                        target_bus = bus
                        break
                else:
                    if bus["CURRENT_NODE_ORD"] in [1, 2, 3]:
                        tracked_plate = bus["PLATE_NO"].strip()
                        target_bus = bus
                        log("trackSingleBus", f"{stdid} 추적 시작: {tracked_plate} (ORD {bus['CURRENT_NODE_ORD']})")

                        if 1 not in reached_ords:
                            reached_ords.add(1)
                            stop_reached_logs.append({"ord": 1, "time": f"{datetime.now().strftime('%Y-%m-%d')} {start_time_str}:00", "note": "ORD 1 출발시간 고정 삽입"})

                        if bus["CURRENT_NODE_ORD"] == 3 and 2 not in reached_ords:
                            now = datetime.now()
                            start_dt = datetime.strptime(f"{now.strftime('%Y-%m-%d')} {start_time_str}:00", "%Y-%m-%d %H:%M:%S")
                            mid_dt = start_dt + (now - start_dt) / 2
                            reached_ords.add(2)
                            stop_reached_logs.append({"ord": 2, "time": mid_dt.strftime("%Y-%m-%d %H:%M:%S"), "note": "ORD 2 중간값 보간 삽입"})
                        break

            if not target_bus:
                log("trackSingleBus", f"{stdid} [대기] 대상 버스 없음")
                if reached_end_minus1 and time.time() - last_movement > 30:
                    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    stop_reached_logs.append({"ord": end_ord, "time": now_time, "note": "종점-1 도달 이후 버스 추적 끊김 → 종점 도달로 간주"})
                    break
                if time.time() - last_movement > 10 * 60:
                    break
                time.sleep(5)
                continue

            now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ord = target_bus["CURRENT_NODE_ORD"]
            lat = target_bus["LAT"]
            lng = target_bus["LNG"]

            matched = get_closest_route_node(lat, lng, ord, node_list, ord_pair_map, last_node_id)
            if matched:
                last_node_id = matched["NODE_ID"]

            if ord != current_ord:
                current_ord = ord
                ord_stay_start = time.time()

            if ord not in reached_ords:
                reached_ords.add(ord)
                stop_reached_logs.append({"ord": ord, "time": now_time})
                last_movement = time.time()

                if ord == end_ord_minus1:
                    reached_end_minus1 = True

            location_logs.append({
                "time": now_time,
                "lat": lat,
                "lng": lng,
                "matched_route_node": matched
            })

            if ord_stay_start and time.time() - ord_stay_start > 10 * 60:
                break

            if ord == end_ord:
                break

            if reached_end_minus1 and end_ord not in reached_ords:
                end_stop = next((s for s in stop_list if s["STOP_ORD"] == end_ord), None)
                if end_stop:
                    dist_to_end = haversine_distance(lat, lng, end_stop["LAT"], end_stop["LNG"])
                    if dist_to_end <= 100:
                        stop_reached_logs.append({"ord": end_ord, "time": now_time, "note": "종점 근접 거리 기반 도달 판정"})
                        break

            time.sleep(10)

    except KeyboardInterrupt:
        log("trackSingleBus", f"{stdid} 수동 중단됨. 로그 저장 중...")

    with open(bus_file_path, "w", encoding="utf-8") as f:
        json.dump({"stop_reached_logs": stop_reached_logs}, f, ensure_ascii=False, indent=2)

    with open(pos_file_path, "w", encoding="utf-8") as f:
        json.dump(location_logs, f, ensure_ascii=False, indent=2)

    log("trackSingleBus", f"{stdid} 저장 완료: stop={bus_file_path}, pos={pos_file_path}")

if __name__ == "__main__":
    stdid = int(sys.argv[1])
    start_time = sys.argv[2]
    log("trackSingleBus", f"STDID {stdid} 진입 성공")
    track_bus(stdid, start_time)