# backend/source/tools/nxnyMapper.py
# grouped_nxny.json에서 각 nx_ny의 대표 좌표를 추출하여 nx_ny_coords.json으로 저장

import os
import sys
import json

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# 파일 경로
GROUPED_PATH = os.path.join(BASE_DIR, "data", "processed", "grouped_nxny.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_coords2.json")

# 데이터 로드
with open(GROUPED_PATH, "r", encoding="utf-8") as f:
    grouped = json.load(f)

coords = {}

# 각 nx_ny 그룹에 대해 첫 번째 노드를 대표로 사용
for nx_ny, nodes in grouped.items():
    first = nodes[0]
    coords[nx_ny] = {
        "lat": first["lat"],
        "lng": first["lng"]
    }

# 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(coords, f, ensure_ascii=False, indent=2)

print(f"대표 좌표 저장 완료: {OUTPUT_PATH}")