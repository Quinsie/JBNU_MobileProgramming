# backend/source/tools/generateRouteNodes.py

import os
import json
import sys
from multiprocessing import Pool
import math

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from source.utils.haversine import haversine_distance

SAMPLE_INTERVAL = 200  # INTERMEDIATE 노드 간격
STOP_MATCH_THRESHOLD = 30  # 정류장 인식 거리
FINE_STEP = 1  # 정류장 감시용 복간 거리 (m)
ANGLE_THRESHOLD = 150  # 까마로 방향 전환 경우 판단 하키는 가장 적절한 각도 (deg)

def interpolate_point(p1, p2, dist_from_p1):
    lat1, lng1 = p1
    lat2, lng2 = p2
    total_dist = haversine_distance(lat1, lng1, lat2, lng2)
    if total_dist == 0:
        return p1
    ratio = dist_from_p1 / total_dist
    lat = lat1 + (lat2 - lat1) * ratio
    lng = lng1 + (lng2 - lng1) * ratio
    return (lat, lng)

def angle_between(p1, p2, p3):
    def to_vec(a, b):
        return (b[0] - a[0], b[1] - a[1])
    def dot(u, v):
        return u[0]*v[0] + u[1]*v[1]
    def norm(v):
        return math.sqrt(v[0]**2 + v[1]**2)

    u = to_vec(p2, p1)
    v = to_vec(p2, p3)
    denom = norm(u) * norm(v)
    if denom == 0:
        return 180
    cos_theta = dot(u, v) / denom
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)

def process_route(stdid):
    VTX_PATH = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "vtx", f"{stdid}.json")
    STOP_PATH = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops", f"{stdid}.json")
    SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "route_nodes", f"{stdid}.json")

    if not os.path.exists(VTX_PATH) or not os.path.exists(STOP_PATH):
        return f"{stdid} skipped"

    with open(VTX_PATH, encoding="utf-8") as f:
        vtx_list = [(pt["LAT"], pt["LNG"]) for pt in json.load(f)["resultList"]]

    with open(STOP_PATH, encoding="utf-8") as f:
        stop_data = json.load(f)["resultList"]
        stop_seq = sorted(stop_data, key=lambda x: x["STOP_ORD"])
        stops = [{"STOP_ID": s["STOP_ID"], "LAT": s["LAT"], "LNG": s["LNG"], "ORD": s["STOP_ORD"]} for s in stop_seq]

    if not stops:
        return f"{stdid} has no stops"

    def find_nearest_idx(target):
        min_d = float("inf")
        min_idx = 0
        for i, (lat, lng) in enumerate(vtx_list):
            d = haversine_distance(lat, lng, target["LAT"], target["LNG"])
            if d < min_d:
                min_d = d
                min_idx = i
        return min_idx

    start_idx = find_nearest_idx(stops[0])
    end_idx = find_nearest_idx(stops[-1])
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    vtx_crop = vtx_list[start_idx:end_idx + 1]
    if len(vtx_crop) < 2:
        print(f"[WARN] {stdid}: VTX too short in cropped range {start_idx}~{end_idx}, using full VTX")
        vtx_crop = vtx_list

    vtx_list = vtx_crop

    output = []
    node_id = 0
    sample_acc = 0.0
    stop_index = 0
    cur_pos = vtx_list[0]
    visited = set()
    last_angle_check = None

    for i in range(1, len(vtx_list)):
        prev = cur_pos
        nxt = vtx_list[i]
        seg_dist = haversine_distance(*prev, *nxt)
        if seg_dist == 0:
            continue

        step = 0.0
        while step < seg_dist:
            interp = interpolate_point(prev, nxt, step)
            step += FINE_STEP

            if last_angle_check:
                ang = angle_between(last_angle_check, cur_pos, interp)
                if ang < ANGLE_THRESHOLD:
                    step += FINE_STEP * 3
                    continue
            last_angle_check = cur_pos
            cur_pos = interp

            key = (round(cur_pos[0], 6), round(cur_pos[1], 6))
            if key in visited:
                continue
            visited.add(key)

            if stop_index < len(stops):
                current_stop = stops[stop_index]
                dist_to_stop = haversine_distance(cur_pos[0], cur_pos[1], current_stop["LAT"], current_stop["LNG"])
                if dist_to_stop < STOP_MATCH_THRESHOLD:
                    output.append({
                        "NODE_ID": node_id,
                        "TYPE": "STOP",
                        "STOP_ID": current_stop["STOP_ID"],
                        "LAT": current_stop["LAT"],
                        "LNG": current_stop["LNG"]
                    })
                    node_id += 1
                    stop_index += 1
                    continue

            sample_acc += FINE_STEP
            if sample_acc >= SAMPLE_INTERVAL:
                output.append({
                    "NODE_ID": node_id,
                    "TYPE": "INTERMEDIATE",
                    "LAT": cur_pos[0],
                    "LNG": cur_pos[1]
                })
                node_id += 1
                sample_acc = 0.0

    for i, node in enumerate(output):
        node["NODE_ID"] = i

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return f"{stdid} done"

def run_all_routes():
    STOP_PATH = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
    stdid_list = [fname.replace(".json", "") for fname in os.listdir(STOP_PATH) if fname.endswith(".json")]

    with Pool() as pool:
        results = pool.map(process_route, stdid_list)

    for r in results:
        print(r)

if __name__ == "__main__":
    run_all_routes()