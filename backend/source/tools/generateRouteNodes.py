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
    vtx_path = os.path.join(VTX_DIR, f"{stdid}.json")
    stop_path = os.path.join(STOP_DIR, f"{stdid}.json")

    if not os.path.exists(stop_path) or not os.path.exists(vtx_path):
        return

    with open(stop_path, encoding="utf-8") as f:
        stop_data = json.load(f)["resultList"]
    with open(vtx_path, encoding="utf-8") as f:
        vtx_data = json.load(f)["resultList"]

    vtx_points = [(v["LAT"], v["LNG"]) for v in vtx_data if v.get("LAT") and v.get("LNG")]
    stop_coords = [(s["LAT"], s["LNG"], s["STOP_ID"]) for s in stop_data if s.get("LAT") and s.get("LNG")]
    stop_points = [
        {"lat": s[0], "lng": s[1], "stop_id": s[2], "type": "stop", **get_nearest_node(s[0], s[1])}
        for s in stop_coords
    ]

    route_nodes = []
    seen = []

    for i in range(len(stop_points) - 1):
        a = stop_points[i]
        b = stop_points[i + 1]
        if not seen or haversine_distance(a["lat"], a["lng"], seen[-1]["lat"], seen[-1]["lng"]) >= 30:
            route_nodes.append(a)
            seen.append(a)

        sub_path = []
        found_a, found_b = None, None
        for idx, (lat, lng) in enumerate(vtx_points):
            if found_a is None and haversine_distance(lat, lng, a["lat"], a["lng"]) < 30:
                found_a = idx
            if haversine_distance(lat, lng, b["lat"], b["lng"]) < 30:
                found_b = idx
                if found_a is not None:
                    break

        if found_a is not None and found_b is not None and found_a < found_b:
            sub_path = vtx_points[found_a:found_b + 1]
        else:
            sub_path = [(a["lat"], a["lng"]), (b["lat"], b["lng"])]

        sampled = []
        dist_accum = 0
        prev = sub_path[0]
        for lat, lng in sub_path[1:]:
            dist = haversine_distance(prev[0], prev[1], lat, lng)
            dist_accum += dist
            if dist_accum >= 50:
                node = {"type": "mid", **get_nearest_node(lat, lng)}
                if not any(haversine_distance(node["lat"], node["lng"], n["lat"], n["lng"]) < 30 for n in seen):
                    sampled.append(node)
                    seen.append(node)
                dist_accum = 0
            prev = (lat, lng)

        route_nodes.extend(sampled)

    last = stop_points[-1]
    if not route_nodes or haversine_distance(last["lat"], last["lng"], route_nodes[-1]["lat"], route_nodes[-1]["lng"]) > 30:
        route_nodes.append(last)

    with open(os.path.join(OUTPUT_DIR, f"{stdid}.json"), "w", encoding="utf-8") as f:
        json.dump(route_nodes, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    stdid_list = [fn.replace(".json", "") for fn in os.listdir(VTX_DIR) if fn.endswith(".json")]
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap(process_single_stdid, stdid_list), total=len(stdid_list)))