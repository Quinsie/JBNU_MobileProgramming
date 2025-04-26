# backend/source/tools/stdidNumberMap.py
# STDID->버스번호/상하행/몇번째노선 인지 매핑해주는 툴툴

import os
import json

from collections import defaultdict

# 경로 설정
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUBLIST_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "subList")
SAVE_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")

# 결과 저장용
stdid_map = {}

# 노선번호+방향별로 등장 순서 카운터
counter = defaultdict(int)

# 모든 subList 파일 순회
for file in os.listdir(SUBLIST_DIR):
    if not file.endswith(".json"):
        continue

    filepath = os.path.join(SUBLIST_DIR, file)
    with open(filepath, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[경고] {file} 로딩 실패, JSON 형식 오류.")
            continue

    result_list = data.get("resultList", [])

    for item in result_list:
        stdid = str(item.get("BRT_STDID"))
        brt_no = item.get("BRT_NO", "").strip()
        direction = item.get("BRT_DIRECTION")

        if not stdid or not brt_no or not direction:
            continue  # 필수 정보 없으면 스킵

        # dash(-)는 언더바(_)로 변환
        brt_no = brt_no.replace("-", "_")

        # 방향 결정
        if str(direction) == "1":
            dir_letter = "A"  # 상행
        elif str(direction) == "2":
            dir_letter = "B"  # 하행
        else:
            dir_letter = "U"  # 알 수 없는 방향 (예외 대응)

        # 현재 노선+방향 조합 카운터 증가
        key = f"{brt_no}{dir_letter}"
        counter[key] += 1

        # 최종 라벨 생성 (ex: 119A1, 119B2)
        label = f"{key}{counter[key]}"

        # 매핑 저장
        stdid_map[stdid] = label

# 결과 저장
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(stdid_map, f, ensure_ascii=False, indent=2)

print(f"STDID 매핑 완료: {len(stdid_map)}개 생성 → {SAVE_PATH}")
