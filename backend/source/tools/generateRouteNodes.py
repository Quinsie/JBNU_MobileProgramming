# backend/source/tools/generateRouteNodes.py

import os
import json
import sys
from tqdm import tqdm

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
VTX_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "vtx")
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
TRAFFIC_NODE_PATH = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "traffic", "20250503_1141.json")  # 최신 파일 선택
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")
from source.utils.haversine import haversine_distance
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 가장 최근 traffic json 불러오기 (파일명 최신순)
with open(os.path.join(TRAFFIC_NODE_PATH), encoding="utf-8") as f:
    traffic_nodes = json.load(f)

# traffic node 좌표만 추출
traffic_node_coords = [
    {"id": t["id"], "sub": t["sub"], "lat": t.get("lat"), "lng": t.get("lng")}
    for t in traffic_nodes
    if t.get("lat") is not None and t.get("lng") is not None
]

# stdid 목록
stdid_list = [fn.replace(".json", "") for fn in os.listdir(VTX_DIR) if fn.endswith(".json")]

# 각 stdid 처리
for stdid in tqdm(stdid_list):
    vtx_path = os.path.join(VTX_DIR, f"{stdid}.json")
    stop_path = os.path.join(STOP_DIR, f"{stdid}.json")

    if not os.path.exists(vtx_path) or not os.path.exists(stop_path):
        continue

    with open(vtx_path, encoding="utf-8") as f:
        vtx_data = json.load(f)["resultList"]

    with open(stop_path, encoding="utf-8") as f:
        stop_data = json.load(f)["resultList"]

    # VTX point list
    vtx_points = [(v["LAT"], v["LNG"]) for v in vtx_data if v.get("LAT") and v.get("LNG")]
    if len(vtx_points) < 2:
        continue

    # 정류장 좌표 목록
    stop_coords = [(s["LAT"], s["LNG"], s["STOP_ID"]) for s in stop_data if s.get("LAT") and s.get("LNG")]

    # 최종 노드 리스트
    route_nodes = []
    included_stops = set()

    for i in range(len(vtx_points) - 1):
        A = vtx_points[i]
        B = vtx_points[i + 1]

        # A, B도 포함
        def get_nearest_node(lat, lng):
            dists = [(haversine_distance(lat, lng, t["lat"], t["lng"]), t) for t in traffic_node_coords]
            return sorted(dists, key=lambda x: x[0])[0][1] if dists else {}

        nearest_A = get_nearest_node(*A)
        if nearest_A:
            route_nodes.append({"type": "vtx", **nearest_A})

        # A→B 선분 기준 가까운 traffic node 3개
        def project_and_filter():
            result = []
            for t in traffic_node_coords:
                ax, ay = A
                bx, by = B
                tx, ty = t["lat"], t["lng"]

                dx, dy = bx - ax, by - ay
                if dx == dy == 0:
                    continue
                t_val = ((tx - ax) * dx + (ty - ay) * dy) / (dx * dx + dy * dy)
                if 0 <= t_val <= 1:
                    dist = haversine_distance(tx, ty, ax + t_val * dx, ay + t_val * dy)
                    result.append((dist, t))
            return [r[1] for r in sorted(result, key=lambda x: x[0])[:3]] if result else []

        segment_nodes = project_and_filter()
        if not segment_nodes:
            print(f"[INFO] No traffic node found between VTX segment {i}-{i+1} for STDID {stdid}, skipping segment.")
        else:
            route_nodes.extend(segment_nodes)

        # 정류장 삽입 (A→B 사이에 존재하는 정류장 추가)
        for lat, lng, stop_id in stop_coords:
            if stop_id in included_stops:
                continue

            def distance_to_segment(px, py, ax, ay, bx, by):
                dx, dy = bx - ax, by - ay
                if dx == dy == 0:
                    return haversine_distance(px, py, ax, ay)
                t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
                proj_x = ax + t * dx
                proj_y = ay + t * dy
                return haversine_distance(px, py, proj_x, proj_y)

            d = distance_to_segment(lat, lng, *A, *B)
            if d < 50:
                nearest_stop = get_nearest_node(lat, lng)
                if nearest_stop:
                    route_nodes.append({"type": "stop", **nearest_stop, "stop_id": stop_id})
                    included_stops.add(stop_id)

    # 마지막 B도 포함
    nearest_B = get_nearest_node(*vtx_points[-1])
    if nearest_B:
        route_nodes.append({"type": "vtx", **nearest_B})

    # 저장
    with open(os.path.join(OUTPUT_DIR, f"{stdid}.json"), "w", encoding="utf-8") as f:
        json.dump(route_nodes, f, indent=2, ensure_ascii=False)
