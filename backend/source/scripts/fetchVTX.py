# backend/source/scripts/fetchVTX.py
# 버스 노선별 경로 중간부분을 ID와 LAT, LONG으로 딴 정보를 제공하는 API로부터 해당 정보를 도로ID, LAT, LONG으로 구성된 정보를 backend/data/raw/staticInfo/vtx/ 아래에 버스 노선 STDID.json별로 저장

import os
import json
import time
import requests

URL = "http://www.jeonjuits.go.kr/bis/selectBisRouteVtxList.do"  # 노선 궤적 추출 API
SUBLIST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "subList"))
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "vtx"))

def fetch_vtx():
    os.makedirs(PATH, exist_ok=True)

    for filename in os.listdir(SUBLIST_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(SUBLIST_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            sublist_data = json.load(f)

        for sub in sublist_data.get("resultList", []):
            stdid = str(sub.get("BRT_STDID"))
            payload = {
                "locale": "ko-kr",
                "routeId": stdid
            }

            try:
                res = requests.post(URL, data=payload, timeout=5)
                res.raise_for_status()
                result = res.json()

                # 🔽 IDX 순번 부여
                for i, pt in enumerate(result.get("resultList", [])):
                    pt["IDX"] = i + 1

                save_path = os.path.join(PATH, f"{stdid}.json")
                with open(save_path, "w", encoding="utf-8") as out:
                    json.dump(result, out, ensure_ascii=False, indent=2)

                print(f"[OK] {stdid} 저장 및 인덱싱 완료")

            except Exception as e:
                print(f"[FAIL] STDID {stdid} | {e}")

            time.sleep(0.3)

if __name__ == "__main__":
    fetch_vtx()