# backend/source/tools/nxnyCalculator.py
# 도로교통정보 노드를 기상청 기준 nx_ny로 묶어서 저장하는 스크립트

import os
import sys
import json

# BASE_DIR 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from source.utils.convertToGrid import convert_to_grid

# 파일 경로
TRAF_PATH = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "traf_vtxlist.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "grouped_nxny.json")

# JSON 로드
with open(TRAF_PATH, "r", encoding="utf-8") as f:
    nodes = json.load(f)

nx_ny_grouped = {}

for node in nodes:
    lat, lng = node["lat"], node["lng"]
    nx, ny = convert_to_grid(lat, lng)
    key = f"{nx}_{ny}"

    if key not in nx_ny_grouped:
        nx_ny_grouped[key] = []
    nx_ny_grouped[key].append(node)

# 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nx_ny_grouped, f, ensure_ascii=False, indent=2)

print(f"저장 완료: {OUTPUT_PATH}")