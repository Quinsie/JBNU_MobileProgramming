# backend/source/tools/mapRouteNodesToTraffic.py

import os
import sys
import json
from math import inf
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import lru_cache

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.haversine import haversine_distance

# 경로 설정
ROUTE_NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")
TRAF_SAMPLE_PATH = os.path.join(BASE_DIR, "data", "processed", "trafficNodesSample.json")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_mapped")
os.makedirs(SAVE_DIR, exist_ok=True)

with open(TRAF_SAMPLE_PATH, encoding="utf-8") as f:
    TRAF_NODES = json.load(f)

@lru_cache(maxsize=None)
def cached_distance(lat1, lng1, lat2, lng2):
    return haversine_distance(lat1, lng1, lat2, lng2)

def find_closest_traffic_node(lat, lng, radius=100):
    closest = None
    min_dist = inf
    for v in TRAF_NODES:
        d = cached_distance(lat, lng, v["lat"], v["lng"])
        if d < min_dist and d <= radius:
            min_dist = d
            closest = {"id": v["id"], "sub": v["sub"], "distance": d}
    return closest

def map_nodes(stdid):
    input_path = os.path.join(ROUTE_NODE_DIR, f"{stdid}.json")
    output_path = os.path.join(SAVE_DIR, f"{stdid}.json")

    with open(input_path, encoding="utf-8") as f:
        nodes = json.load(f)

    matched = [None] * len(nodes)

    # 1차 매핑: 반경 내 가장 가까운 traffic node
    for i, node in enumerate(nodes):
        match = find_closest_traffic_node(node["LAT"], node["LNG"])
        if match:
            matched[i] = match

    # 2차 전파: 인접 노드로부터 매핑 복사
    for i in range(len(nodes)):
        if matched[i] is None:
            left = matched[i - 1] if i - 1 >= 0 and matched[i - 1] and "id" in matched[i - 1] else None
            right = matched[i + 1] if i + 1 < len(nodes) and matched[i + 1] and "id" in matched[i + 1] else None
            if left and right:
                matched[i] = left if left["distance"] <= right["distance"] else right
            elif left:
                matched[i] = left
            elif right:
                matched[i] = right

    # 결과 저장
    result = []
    for i, node in enumerate(nodes):
        new_node = {
            "NODE_ID": node["NODE_ID"],
            "LAT": node["LAT"],
            "LNG": node["LNG"],
            "TYPE": node["TYPE"]
        }
        if "STOP_ID" in node:
            new_node["STOP_ID"] = node["STOP_ID"]
        new_node["matched"] = matched[i] if matched[i] else None
        result.append(new_node)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"resultList": result}, f, ensure_ascii=False, indent=2)

def main():
    files = [f for f in os.listdir(ROUTE_NODE_DIR) if f.endswith(".json")]
    stdids = [f.replace(".json", "") for f in files]
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(map_nodes, stdids), total=len(stdids)))

if __name__ == "__main__":
    main()