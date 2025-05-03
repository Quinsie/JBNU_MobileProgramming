# backend/source/tools/generateRouteNodes.py

import os
import json
import sys
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
VTX_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "vtx")
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
TRAFFIC_NODE_PATH = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "traffic", "20250503_2000.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")
from source.utils.haversine import haversine_distance
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 트래픽 노드 불러오기
with open(os.path.join(TRAFFIC_NODE_PATH), encoding="utf-8") as f:
    traffic_nodes = json.load(f)

traffic_node_coords = [
    {"id": t["id"], "sub": t["sub"], "lat": t.get("lat"), "lng": t.get("lng")}
    for t in traffic_nodes
    if t.get("lat") is not None and t.get("lng") is not None
]

# 유틸 함수
def get_nearest_node(lat, lng, radius=300):
    dists = [(haversine_distance(lat, lng, t["lat"], t["lng"]), t) for t in traffic_node_coords]
    dists = [item for item in dists if item[0] <= radius]
    return sorted(dists, key=lambda x: x[0])[0][1] if dists else None

def distance_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return haversine_distance(px, py, ax, ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return haversine_distance(px, py, proj_x, proj_y)

def process_single_stdid(stdid):
    vtx_path = os.path.join(VTX_DIR, f"{stdid}.json")
    stop_path = os.path.join(STOP_DIR, f"{stdid}.json")

    if not os.path.exists(vtx_path) or not os.path.exists(stop_path):
        return

    with open(vtx_path, encoding="utf-8") as f:
        vtx_data = json.load(f)["resultList"]
    with open(stop_path, encoding="utf-8") as f:
        stop_data = json.load(f)["resultList"]

    vtx_points = []
    seen = set()
    for v in vtx_data:
        key = (v["LAT"], v["LNG"])
        if v.get("LAT") and v.get("LNG") and key not in seen:
            vtx_points.append(key)
            seen.add(key)
    if len(vtx_points) < 2:
        return

    stop_coords = [(s["LAT"], s["LNG"], s["STOP_ID"]) for s in stop_data if s.get("LAT") and s.get("LNG")]

    all_nodes = []
    included_stops = set()
    stop_nodes = []

    for i in range(len(vtx_points) - 1):
        A = vtx_points[i]
        B = vtx_points[i + 1]

        for point in [A, B]:
            node = get_nearest_node(*point)
            if node:
                all_nodes.append({"type": "vtx", **node})
            else:
                all_nodes.append({"type": "vtx", "lat": point[0], "lng": point[1], "id": None, "sub": None})

        for j in range(1, 4):
            ratio = j / 4
            px = A[0] + (B[0] - A[0]) * ratio
            py = A[1] + (B[1] - A[1]) * ratio
            node = get_nearest_node(px, py)
            if node:
                all_nodes.append({"type": "mid", **node})
            else:
                all_nodes.append({"type": "mid", "lat": px, "lng": py, "id": None, "sub": None})

        for lat, lng, stop_id in stop_coords:
            if stop_id in included_stops:
                continue
            d = distance_to_segment(lat, lng, *A, *B)
            if d < 70:
                node = get_nearest_node(lat, lng)
                if node:
                    stop_nodes.append({"type": "stop", **node, "stop_id": stop_id})
                else:
                    stop_nodes.append({"type": "stop", "lat": lat, "lng": lng, "id": None, "sub": None, "stop_id": stop_id})
                included_stops.add(stop_id)

    # 정류장-정류장 기준 필터링
    final_nodes = []
    current_segment = []
    for node in all_nodes:
        current_segment.append(node)
        if node["type"] == "stop":
            deduped = []
            for n in current_segment:
                if n["type"] == "stop" or all(haversine_distance(n["lat"], n["lng"], d["lat"], d["lng"]) >= 50 for d in deduped):
                    deduped.append(n)
            final_nodes.extend(deduped)
            current_segment = [node]

    if current_segment:
        final_nodes.extend(current_segment)

    final_nodes.extend(stop_nodes)

    with open(os.path.join(OUTPUT_DIR, f"{stdid}.json"), "w", encoding="utf-8") as f:
        json.dump(final_nodes, f, indent=2, ensure_ascii=False)

# 멀티프로세싱 실행
if __name__ == "__main__":
    stdid_list = [fn.replace(".json", "") for fn in os.listdir(VTX_DIR) if fn.endswith(".json")]
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_single_stdid, stdid_list), total=len(stdid_list)))
