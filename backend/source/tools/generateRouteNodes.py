# backend/source/tools/generateRouteNodes.py

import os
import json
import sys
from multiprocessing import Pool

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from source.utils.haversine import haversine_distance

SAMPLE_INTERVAL = 200  # INTERMEDIATE 노드 간격
STOP_MATCH_THRESHOLD = 30  # 정류장 인식 거리
FINE_STEP = 1  # 정류장 감시용 보간 간격 (단위: meter)


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
    current_stop = stops[stop_index] if stop_index < len(stops) else None
    detected_stop_ids = set()
    cur_pos = vtx_list[0]

    for i in range(1, len(vtx_list)):
        prev = cur_pos
        nxt = vtx_list[i]
        seg_dist = haversine_distance(*prev, *nxt)
        if seg_dist == 0:
            seg_dist = 0.001

        step = 0.0
        while step < seg_dist:
            interp = interpolate_point(prev, nxt, step)
            step += FINE_STEP
            cur_pos = interp

            while current_stop:
                dist_to_stop = haversine_distance(cur_pos[0], cur_pos[1], current_stop["LAT"], current_stop["LNG"])
                if dist_to_stop < STOP_MATCH_THRESHOLD and current_stop["STOP_ID"] not in detected_stop_ids:
                    output.append({
                        "NODE_ID": node_id,
                        "TYPE": "STOP",
                        "STOP_ID": current_stop["STOP_ID"],
                        "LAT": cur_pos[0],
                        "LNG": cur_pos[1]
                    })
                    node_id += 1
                    detected_stop_ids.add(current_stop["STOP_ID"])
                    stop_index += 1
                    current_stop = stops[stop_index] if stop_index < len(stops) else None
                else:
                    break

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

    for stop in stops:
        if stop["STOP_ID"] not in detected_stop_ids:
            min_d = float("inf")
            insert_idx = 0
            for j, node in enumerate(output):
                d = haversine_distance(node["LAT"], node["LNG"], stop["LAT"], stop["LNG"])
                if d < min_d:
                    min_d = d
                    insert_idx = j

            output.insert(insert_idx, {
                "NODE_ID": None,
                "TYPE": "STOP",
                "STOP_ID": stop["STOP_ID"],
                "LAT": stop["LAT"],
                "LNG": stop["LNG"]
            })
            print(f"[FIXED] {stdid}: inserted STOP {stop['STOP_ID']} at {insert_idx}")

    # 시점 STOP이 첫 번째에 없으면 앞에 추가
    if output[0]["TYPE"] != "STOP" or output[0]["STOP_ID"] != stops[0]["STOP_ID"]:
        output.insert(0, {
            "NODE_ID": None,
            "TYPE": "STOP",
            "STOP_ID": stops[0]["STOP_ID"],
            "LAT": stops[0]["LAT"],
            "LNG": stops[0]["LNG"]
        })
        print(f"[FIXED] {stdid}: forced insert of first STOP {stops[0]['STOP_ID']} at HEAD")

    # 종점 STOP이 마지막에 없으면 뒤에 추가
    if output[-1]["TYPE"] != "STOP" or output[-1]["STOP_ID"] != stops[-1]["STOP_ID"]:
        output.append({
            "NODE_ID": None,
            "TYPE": "STOP",
            "STOP_ID": stops[-1]["STOP_ID"],
            "LAT": stops[-1]["LAT"],
            "LNG": stops[-1]["LNG"]
        })
        print(f"[FIXED] {stdid}: forced insert of last STOP {stops[-1]['STOP_ID']} at TAIL")

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