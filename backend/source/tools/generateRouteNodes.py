# backend/source/tools/generateRouteNodes.py

import os
import sys
import json
from multiprocessing import Pool


# 기본 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from source.utils.haversine import haversine_distance
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
VTX_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "vtx")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")

SSTOP_MATCH_THRESHOLD = 100  # 범위 확장
SAMPLE_INTERVAL = 200
FINE_STEP = 1


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


def process_route(stdid):
    vtx_path = os.path.join(VTX_DIR, f"{stdid}.json")
    stop_path = os.path.join(STOP_DIR, f"{stdid}.json")
    save_path = os.path.join(SAVE_DIR, f"{stdid}.json")

    if not os.path.exists(vtx_path) or not os.path.exists(stop_path):
        return f"[SKIP] {stdid} - 파일 없음"

    with open(vtx_path, encoding="utf-8") as f:
        vtx = [(pt["LAT"], pt["LNG"]) for pt in json.load(f)["resultList"]]

    with open(stop_path, encoding="utf-8") as f:
        stops = sorted(json.load(f)["resultList"], key=lambda x: x["STOP_ORD"])

    if len(stops) < 2 or len(vtx) < 2:
        return f"[SKIP] {stdid} - 데이터 부족"

    output, node_id = [], 0
    visited = set()
    stop_index = 0
    acc_dist = 0.0
    cur_pos = vtx[0]

    def add_stop_node(stop):
        nonlocal node_id
        output.append({
            "NODE_ID": node_id,
            "TYPE": "STOP",
            "STOP_ID": stop["STOP_ID"],
            "LAT": stop["LAT"],
            "LNG": stop["LNG"]
        })
        node_id += 1

    def add_intermediate_node(pos):
        nonlocal node_id
        output.append({
            "NODE_ID": node_id,
            "TYPE": "INTERMEDIATE",
            "LAT": pos[0],
            "LNG": pos[1]
        })
        node_id += 1

    for i in range(1, len(vtx)):
        prev, nxt = cur_pos, vtx[i]
        seg_dist = haversine_distance(*prev, *nxt)
        if seg_dist == 0:
            continue

        d = 0
        while d < seg_dist:
            interp = interpolate_point(prev, nxt, d)
            d += FINE_STEP
            cur_pos = interp

            # STOP 감지
            if stop_index < len(stops):
                s = stops[stop_index]
                dist = haversine_distance(interp[0], interp[1], s["LAT"], s["LNG"])
                if dist < STOP_MATCH_THRESHOLD:
                    add_stop_node(s)  # 정류장 위치 기준으로 노드 추가
                    stop_index += 1
                    continue

            acc_dist += FINE_STEP
            if acc_dist >= SAMPLE_INTERVAL:
                acc_dist = 0
                key = (round(interp[0], 6), round(interp[1], 6))
                if key not in visited:
                    visited.add(key)
                    add_intermediate_node(interp)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return f"[OK] {stdid} - {len(output)} nodes"


def run_all_routes():
    stdid_list = [f.replace(".json", "") for f in os.listdir(STOP_DIR) if f.endswith(".json")]
    with Pool() as pool:
        results = pool.map(process_route, stdid_list)

    for r in results:
        print(r)


if __name__ == "__main__":
    run_all_routes()