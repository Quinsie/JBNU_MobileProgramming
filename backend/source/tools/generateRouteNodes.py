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

def process_single_stdid(stdid):
    stop_path = os.path.join(STOP_DIR, f"{stdid}.json")
    if not os.path.exists(stop_path):
        return

    with open(stop_path, encoding="utf-8") as f:
        stop_data = json.load(f)["resultList"]

    stop_coords = [(s["LAT"], s["LNG"], s["STOP_ID"]) for s in stop_data if s.get("LAT") and s.get("LNG")]
    stop_points = [
        {"lat": s[0], "lng": s[1], "stop_id": s[2], "type": "stop", **get_nearest_node(s[0], s[1])}
        for s in stop_coords
    ]

    route_nodes = []
    for i in range(len(stop_points) - 1):
        a = stop_points[i]
        b = stop_points[i + 1]
        route_nodes.append(a)

        dist = haversine_distance(a["lat"], a["lng"], b["lat"], b["lng"])
        num_samples = 0
        if dist > 180:
            num_samples = 3
        elif dist > 120:
            num_samples = 2
        elif dist > 60:
            num_samples = 1

        for j in range(1, num_samples + 1):
            ratio = j / (num_samples + 1)
            px = a["lat"] + (b["lat"] - a["lat"]) * ratio
            py = a["lng"] + (b["lng"] - a["lng"]) * ratio
            node = get_nearest_node(px, py)
            route_nodes.append({"type": "mid", **node})

    route_nodes.append(stop_points[-1])

    final_nodes = []
    for node in route_nodes:
        is_duplicate = False
        for existing in final_nodes:
            if haversine_distance(node["lat"], node["lng"], existing["lat"], existing["lng"]) < 50:
                if existing["type"] == "stop":
                    continue
                if node["id"] is not None and existing["id"] is None:
                    final_nodes.remove(existing)
                    break
                is_duplicate = True
                break
        if not is_duplicate:
            final_nodes.append(node)

    with open(os.path.join(OUTPUT_DIR, f"{stdid}.json"), "w", encoding="utf-8") as f:
        json.dump(final_nodes, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    stdid_list = [fn.replace(".json", "") for fn in os.listdir(VTX_DIR) if fn.endswith(".json")]
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_single_stdid, stdid_list), total=len(stdid_list)))