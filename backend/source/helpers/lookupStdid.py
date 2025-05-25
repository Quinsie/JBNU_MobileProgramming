# backend/source/helpers/lookup_stdid.py

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SUBLIST_DIR = BASE_DIR / "data" / "raw" / "staticInfo" / "subList"
STDID_NUMBER_PATH = BASE_DIR / "data" / "processed" / "stdid_number.json"

# 모든 JSON 파일을 순회하며 STDID → 노선번호(BRT_NO) 매핑 생성
stdid_to_bus = {}

for file in SUBLIST_DIR.glob("*.json"):
    with open(file, encoding="utf-8") as f:
        try:
            data = json.load(f)
            for item in data.get("resultList", []):
                stdid = item.get("BRT_STDID")
                bus_no = item.get("BRT_NO")
                if stdid and bus_no:
                    stdid_to_bus[stdid] = bus_no
        except Exception as e:
            print(f"{file.name} 에러 발생: {e}")

# STDID 번호 매핑 파일 로드
try:
    with open(STDID_NUMBER_PATH, encoding="utf-8") as f:
        stdid_number_map = json.load(f)
except Exception as e:
    print(f"stdid_number.json 로드 실패: {e}")
    stdid_number_map = {}

# 입력 루프
print("🔍 STDID 입력하면 버스 번호를 알려드립니다. 종료하려면 Ctrl+C 누르세요.")
while True:
    try:
        stdid = int(input("STDID: "))
        bus_no = stdid_to_bus.get(stdid)
        mapped_info = stdid_number_map.get(str(stdid))

        if bus_no and mapped_info:
            # mapped_info는 예를 들면 "119A1" 이런 형태
            # 예를 들어 54B1 -> 54 / 하행 / 1번째
            base = ''.join([c for c in mapped_info if c.isdigit() or c == '_'])
            if "A" in mapped_info:
                direction = "상행"
                n = mapped_info.split("A")[-1]
            elif "B" in mapped_info:
                direction = "하행"
                n = mapped_info.split("B")[-1]
            else:
                direction = "방향 알 수 없음"
                n = "?"

            print(f"버스 번호: {bus_no} / {direction} {n}번째")
        elif bus_no:
            print(f"버스 번호: {bus_no} / (추가 정보 없음)")
        else:
            print("일치하는 STDID 없음")
    except ValueError:
        print("숫자로 입력해 주세요.")
    except KeyboardInterrupt:
        print("\n종료합니다.")
        break