# backend/source/tools/generateRouteNodes.py

import os
import json
import sys
from multiprocessing import Pool

# haversine.py를 import할 수 있도록 경로 추가
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
from source.utils.haversine import haversine_distance

# 설정
SAMPLE_INTERVAL = 100  # meters

# 두 점 사이 보간 함수
def interpolate_point(p1, p2, target_dist):
    lat1, lng1 = p1
    lat2, lng2 = p2
    total_dist = haversine_distance(lat1, lng1, lat2, lng2)
    if total_dist == 0:
        return p1
    ratio = target_dist / total_dist
    lat = lat1 + (lat2 - lat1) * ratio
    lng = lng1 + (lng2 - lng1) * ratio
    return (lat, lng)

# 하나의 STDID 처리
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
        stops = [{"STOP_ID": s["STOP_ID"], "LAT": s["LAT"], "LNG": s["LNG"]} for s in stop_seq]

    output = []
    node_id = 0
    acc_dist = 0
    i = 0
    cur_pos = vtx_list[0]

    stop_index = 0
    current_stop = stops[stop_index] if stops else None
    min_dist = float("inf")
    best_pos = None

    while i < len(vtx_list) - 1:
        next_pos = vtx_list[i + 1]
        seg_dist = haversine_distance(*cur_pos, *next_pos)

        # 샘플 노드 생성
        if acc_dist + seg_dist >= SAMPLE_INTERVAL:
            remain = SAMPLE_INTERVAL - acc_dist
            new_node = interpolate_point(cur_pos, next_pos, remain)
            output.append({
                "NODE_ID": node_id,
                "TYPE": "INTERMEDIATE",
                "LAT": new_node[0],
                "LNG": new_node[1]
            })
            node_id += 1
            cur_pos = new_node
            acc_dist = 0
            continue  # 다음 segment 이동 없이 다시 current에서 진행

        # 정류장 거리 체크
        if current_stop:
            dist_to_stop = haversine_distance(cur_pos[0], cur_pos[1], current_stop["LAT"], current_stop["LNG"])
            
            if dist_to_stop < min_dist and dist_to_stop < 100:  # 100m 이하에서만 유효
                min_dist = dist_to_stop
                best_pos = (cur_pos[0], cur_pos[1])
            elif best_pos is not None:
                output.append({
                    "NODE_ID": node_id,
                    "TYPE": "STOP",
                    "STOP_ID": current_stop["STOP_ID"],
                    "LAT": best_pos[0],
                    "LNG": best_pos[1]
                })
                node_id += 1
                stop_index += 1
                current_stop = stops[stop_index] if stop_index < len(stops) else None
                min_dist = float("inf")
                best_pos = None

        acc_dist += seg_dist
        cur_pos = next_pos
        i += 1

    # 저장
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return f"{stdid} done"

# 전체 노선 병렬 처리
def run_all_routes():
    STOP_PATH = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
    stdid_list = [fname.replace(".json", "") for fname in os.listdir(STOP_PATH) if fname.endswith(".json")]

    with Pool() as pool:
        results = pool.map(process_route, stdid_list)

    for r in results:
        print(r)

# 진입점
if __name__ == "__main__":
    run_all_routes()