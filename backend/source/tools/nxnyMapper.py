# backend/source/tools/nxnyMapper.py
# grouped_nxny.json에서 각 nx_ny의 대표 좌표를 추출하여 nx_ny_coords.json으로 저장

import os
import sys
import json

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

GROUPED_PATH = os.path.join(BASE_DIR, "data", "processed", "grouped_nxny.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_coords.json")

# 데이터 로드
with open(GROUPED_PATH, "r", encoding="utf-8") as f:
    grouped = json.load(f)

# 단순 {"nx_ny": true} 구조 생성
coords = {key: True for key in grouped}

# 저장
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(coords, f, ensure_ascii=False, indent=2)

print(f"[완료] nx_ny_coords.json 저장 완료: {OUTPUT_PATH}")