# backend/source/tools/lookup_stdid.py

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SUBLIST_DIR = BASE_DIR / "data" / "raw" / "staticInfo" / "subList"

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

# 입력 루프
print("🔍 STDID 입력하면 버스 번호를 알려드립니다. 종료하려면 Ctrl+C 누르세요.")
while True:
    try:
        stdid = int(input("STDID: "))
        bus_no = stdid_to_bus.get(stdid)
        if bus_no:
            print(f"버스 번호: {bus_no}")
        else:
            print("일치하는 STDID 없음")
    except ValueError:
        print("숫자로 입력해 주세요.")
    except KeyboardInterrupt:
        print("\n종료합니다.")
        break