import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STOP_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")

def check_route_nodes(stdid):
    stop_path = os.path.join(STOP_DIR, f"{stdid}.json")
    node_path = os.path.join(NODE_DIR, f"{stdid}.json")

    if not os.path.exists(stop_path) or not os.path.exists(node_path):
        print(f"[SKIP] {stdid} - Missing data")
        return

    with open(stop_path, encoding="utf-8") as f:
        stop_data = json.load(f)["resultList"]
        stop_dict = {s["STOP_ID"]: s["STOP_ORD"] for s in stop_data}

    with open(node_path, encoding="utf-8") as f:
        node_data = json.load(f)
        detected_stop_ids = {n["STOP_ID"] for n in node_data if n["TYPE"] == "STOP"}

    missing = []
    for sid, ord in stop_dict.items():
        if sid not in detected_stop_ids:
            missing.append((sid, ord))

    if missing:
        print(f"[MISSING] {stdid} - {len(missing)} stops missing:")
        for sid, ord in sorted(missing, key=lambda x: x[1]):
            print(f"  - STOP_ID: {sid} (ORD: {ord})")


def run_check():
    stdids = [f.replace(".json", "") for f in os.listdir(STOP_DIR) if f.endswith(".json")]
    for stdid in sorted(stdids):
        check_route_nodes(stdid)

if __name__ == "__main__":
    run_check()
