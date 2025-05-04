# backend/source/tools/nxnyRouteGenerator.py
# route_node들을 기상청 nx_ny 기준으로 그룹핑

import os
import sys
import json

# BASE 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.convertToGrid import convert_to_grid

# 경로 설정
ROUTE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "grouped_nxny.json")

# 출력용 딕셔너리
nx_ny_grouped = {}

# 모든 STDID 파일 반복
for filename in os.listdir(ROUTE_DIR):
    if not filename.endswith(".json"):
        continue

    stdid = filename.replace(".json", "")
    file_path = os.path.join(ROUTE_DIR, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    for node in nodes:
        lat = node["LAT"]
        lng = node["LNG"]
        node_id = node["NODE_ID"]

        nx, ny = convert_to_grid(lat, lng)
        key = f"{nx}_{ny}"

        if key not in nx_ny_grouped:
            nx_ny_grouped[key] = []

        nx_ny_grouped[key].append({
            "STDID": stdid,
            "NODE_ID": node_id
        })

# 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nx_ny_grouped, f, ensure_ascii=False, indent=2)

print(f"[완료] route_node 기반 grouped_nxny.json 저장: {OUTPUT_PATH}")