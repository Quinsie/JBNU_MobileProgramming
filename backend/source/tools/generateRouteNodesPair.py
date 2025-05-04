import os
import sys
import json

# BASE_DIR 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# 경로 설정
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")
PAIR_SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_pair")

os.makedirs(PAIR_SAVE_DIR, exist_ok=True)

def round_coord(coord):
    return round(coord, 6)

def validate_stop_nodes(stdid, route_nodes):
    stop_path = os.path.join(STOP_DIR, f"{stdid}.json")
    try:
        with open(stop_path, encoding="utf-8") as f:
            stops = json.load(f)["resultList"]
    except Exception as e:
        print(f"[ERROR] {stdid} 정류장 파일 불러오기 실패: {e}")
        return False

    route_stops = [n for n in route_nodes if n["TYPE"] == "STOP"]

    if len(route_stops) != len(stops):
        print(f"[ERROR] {stdid}: STOP 개수 불일치 (route_nodes: {len(route_stops)} vs stops: {len(stops)})")
        return False

    for i, (node, stop) in enumerate(zip(route_stops, stops)):
        if (round_coord(node["LAT"]) != round_coord(stop["LAT"]) or
            round_coord(node["LNG"]) != round_coord(stop["LNG"]) or
            node.get("STOP_ID") != stop.get("STOP_ID")):
            print(f"[MISMATCH] {stdid} - ORD {i+1}: NODE_ID {node['NODE_ID']} != STOP_ID {stop['STOP_ID']}")
            return False

    return True

def generate_ord_pairs(stdid):
    path = os.path.join(NODE_DIR, f"{stdid}.json")
    try:
        with open(path, encoding="utf-8") as f:
            route_nodes = json.load(f)
    except Exception as e:
        print(f"[ERROR] {stdid} 노드 파일 불러오기 실패: {e}")
        return None

    if not validate_stop_nodes(stdid, route_nodes):
        print(f"[INVALID] {stdid} 유효성 검사 실패 → 무시됨")
        return None

    ord_pairs = {}
    current_ord = 1
    current_range = []

    for node in route_nodes:
        current_range.append(node["NODE_ID"])
        if node["TYPE"] == "STOP":
            if current_ord >= 2:
                ord_pairs[current_ord - 1] = current_range[:-1]  # 현재 STOP 전까지가 이전 ORD 영역
            current_range = [node["NODE_ID"]]  # 현재 STOP부터 시작
            current_ord += 1

    # 마지막 구간도 추가
    if current_range:
        ord_pairs[current_ord - 1] = current_range

    return ord_pairs

def main():
    stdids = [f.split(".")[0] for f in os.listdir(NODE_DIR) if f.endswith(".json")]
    total = len(stdids)
    success = 0

    for i, stdid in enumerate(stdids, 1):
        ord_pairs = generate_ord_pairs(stdid)
        if ord_pairs is None:
            continue
        save_path = os.path.join(PAIR_SAVE_DIR, f"{stdid}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(ord_pairs, f, ensure_ascii=False, indent=2)
        print(f"[{i}/{total}] {stdid} 처리 완료 ✅")
        success += 1

    print(f"\n[완료] 총 {success}개 노선 처리됨 (전체 {total}개)")

if __name__ == "__main__":
    main()