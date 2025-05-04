# backend/source/tools/nxnyLookupGenerator.py
# route_nodes/{STDID}.json 기반으로 nx_ny_lookup.json 생성

import os
import sys
import json

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.convertToGrid import convert_to_grid

ROUTE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_lookup.json")

lookup = {}

# 각 route_node 파일 순회
for filename in os.listdir(ROUTE_DIR):
    if not filename.endswith(".json"):
        continue

    stdid = filename.replace(".json", "")
    file_path = os.path.join(ROUTE_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    for node in nodes:
        node_id = node["NODE_ID"]
        lat = node["LAT"]
        lng = node["LNG"]

        nx, ny = convert_to_grid(lat, lng)
        nx_ny = f"{nx}_{ny}"
        key = f"{stdid}_{node_id}"

        lookup[key] = nx_ny

# 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(lookup, f, ensure_ascii=False, indent=2)

print(f"[완료] nx_ny_lookup.json 저장 완료: {OUTPUT_PATH}")