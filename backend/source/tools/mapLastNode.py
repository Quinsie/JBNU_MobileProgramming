# backend/source/tools/mapLastNode.py

import os
import sys
import json

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
ROUTE_NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "last_node.json")

def extract_last_node_id(file_path):
    with open(file_path, encoding='utf-8') as f:
        data = json.load(f)
        if not data:
            return None
        return data[-1].get("NODE_ID")

if __name__ == "__main__":
    result = {}
    for fname in os.listdir(ROUTE_NODE_DIR):
        if not fname.endswith(".json"):
            continue
        stdid = fname.replace(".json", "")
        path = os.path.join(ROUTE_NODE_DIR, fname)
        last_node_id = extract_last_node_id(path)
        if last_node_id is not None:
            result[stdid] = last_node_id

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {SAVE_PATH}")