# backend/source/tools/mapVtxToRoadId.py
# 6과 7에서 저장한 정보를 기반으로, 7의 정보를 6에 맞춰 매핑한 뒤 backend/data/vtx_mapped/ 에 노선STDID.json별로 저장.

import json
import os
from math import sqrt

# 경로 설정
traf_vtx_path = "backend/data/raw/staticInfo/traf_vtxlist.json"
vtx_dir = "backend/data/raw/staticInfo/vtx"  # 여기를 실제 네 구조에 맞춰서 조정해줘
output_dir = "backend/data/processed/vtx_mapped"  # 결과 저장할 디렉토리

os.makedirs(output_dir, exist_ok=True)

# 거리 계산 함수
def distance(a, b):
    return sqrt((a["lat"] - b["lat"])**2 + (a["lng"] - b["lng"])**2)

# traf_vtxlist 로드
with open(traf_vtx_path, "r", encoding="utf-8") as f:
    traf_vtx_list = json.load(f)

print(f"총 traf_vtx 항목 수: {len(traf_vtx_list)}")

# vtx 디렉토리 순회
for file in os.listdir(vtx_dir):
    if not file.endswith(".json"):
        continue

    with open(os.path.join(vtx_dir, file), "r", encoding="utf-8") as f:
        data = json.load(f)

    mapped_result = []
    for pt in data["resultList"]:
        lng, lat = pt["LNG"], pt["LAT"]
        closest = min(traf_vtx_list, key=lambda v: distance({"lat": lat, "lng": lng}, v))
        pt["MATCHED_ID"] = closest["id"]
        pt["MATCHED_SUB"] = closest["sub"]
        mapped_result.append(pt)

    # 저장
    out_path = os.path.join(output_dir, file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"resultList": mapped_result}, f, ensure_ascii=False, indent=2)

    print(f"[✔] {file} 매핑 완료")

print("🔧 전체 매핑 작업 완료.")