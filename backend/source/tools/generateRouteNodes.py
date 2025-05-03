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

def get_nearest_node(lat, lng, radius=100):
    dists = [(haversine_distance(lat, lng, t["lat"], t["lng"]), t) for t in traffic_node_coords]
    dists = [item for item in dists if item[0] <= radius]
    return sorted(dists, key=lambda x: x[0])[0][1] if dists else {"id": None, "sub": None, "lat": lat, "lng": lng}

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
    if not os.path.exists(vtx_path) or not os.path.exists(stop_path): return

    with open(vtx_path, encoding="utf-8") as f:
        vtx_data = json.load(f)["resultList"]
    with open(stop_path, encoding="utf-8") as f:
        stop_data = json.load(f)["resultList"]

    stop_coords = [(s["LAT"], s["LNG"], s["STOP_ID"]) for s in stop_data if s.get("LAT") and s.get("LNG")]
    stop_points = [{"lat": s[0], "lng": s[1], "stop_id": s[2], "type": "stop", **get_nearest_node(s[0], s[1])} for s in stop_coords]

    mid_nodes_per_segment = []
    for i in range(len(stop_coords) - 1):
        a = stop_coords[i]
        b = stop_coords[i + 1]
        dist = haversine_distance(a[0], a[1], b[0], b[1])

        num_samples = 0
        if dist > 180:
            num_samples = 3
        elif dist > 120:
            num_samples = 2
        elif dist > 60:
            num_samples = 1

        segment_mid_nodes = []
        for j in range(1, num_samples + 1):
            ratio = j / (num_samples + 1)
            px = a[0] + (b[0] - a[0]) * ratio
            py = a[1] + (b[1] - a[1]) * ratio
            node = get_nearest_node(px, py)
            segment_mid_nodes.append({"type": "mid", **node})
        mid_nodes_per_segment.append(segment_mid_nodes)

    filtered_nodes = []
    for i in range(len(stop_points) - 1):
        a = stop_points[i]
        b = stop_points[i + 1]
        segment_nodes = [a] + mid_nodes_per_segment[i] + [b]

        cleaned = []
        for node in segment_nodes:
            is_dup = False
            for f in cleaned:
                if haversine_distance(node["lat"], node["lng"], f["lat"], f["lng"]) < 50:
                    if f["type"] == "stop":
                        continue  # 정류장은 절대 제거하지 않음
                    if node["id"] is not None and f["id"] is None:
                        cleaned.remove(f)
                        break
                    else:
                        is_dup = True
                        break
            if not is_dup:
                cleaned.append(node)
        filtered_nodes.extend(cleaned)

    with open(os.path.join(OUTPUT_DIR, f"{stdid}.json"), "w", encoding="utf-8") as f:
        json.dump(filtered_nodes, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    stdid_list = [fn.replace(".json", "") for fn in os.listdir(VTX_DIR) if fn.endswith(".json")]
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_single_stdid, stdid_list), total=len(stdid_list)))