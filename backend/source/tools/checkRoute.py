import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")

def check_route_nodes(stdid):
    stop_path = os.path.join(STOP_DIR, f"{stdid}.json")
    node_path = os.path.join(NODE_DIR, f"{stdid}.json")

    if not os.path.exists(stop_path) or not os.path.exists(node_path):
        return None

    with open(stop_path, encoding="utf-8") as f:
        stop_data = json.load(f)["resultList"]
        stop_dict = {s["STOP_ID"]: s["STOP_ORD"] for s in stop_data}

    with open(node_path, encoding="utf-8") as f:
        node_data = json.load(f)
        detected_stop_ids = {n["STOP_ID"] for n in node_data if n["TYPE"] == "STOP"}

    missing_count = sum(1 for sid in stop_dict if sid not in detected_stop_ids)
    return (stdid, missing_count)

def run_check():
    stdids = [f.replace(".json", "") for f in os.listdir(STOP_DIR) if f.endswith(".json")]
    results = [check_route_nodes(stdid) for stdid in sorted(stdids)]
    results = [r for r in results if r and r[1] > 0]

    print(f"\n누락된 노선 수: {len(results)}개")
    for stdid, count in results:
        print(f"[MISSING] {stdid} - {count} stops missing")

if __name__ == "__main__":
    run_check()