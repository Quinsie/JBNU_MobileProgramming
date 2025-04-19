# backend/source/scripts/fetchRouteId.py

import os
import json
import requests

URL = "http://www.jeonjuits.go.kr/bis/selectGrpRouteList.do" # 버스 전체노선 API
payload = {"locale": "ko-kr"}
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "routes.json"))

def fetchRoutes():
    try:
        res = requests.post(URL, data = payload, timeout = 5) # 타임아웃 5씩 주면서 데이터 요청
        res.raise_for_status()
        json_data = res.json()
        
        # Response가 "resultList" 하위에 이어서 온다. "BRT_NO" (노선명)만 필요하니까, 받아와서 sort.
        route_num = set()
        for item in json_data.get("resultList", []): route_num.add(item.get("BRT_NO"))
        route_num = list(route_num)
        route_num.sort()

        os.makedirs(os.path.dirname(PATH), exist_ok = True)
        with open(PATH, "w", encoding = "utf-8") as file:
            # 한글 허용을 위해 ensure ascii, indent는 들여쓰기 깊이. 2가 통상적.
            json.dump(route_num, file, ensure_ascii = False, indent = 2)
        
        print("저장 완료")

    except Exception as error:
        print("저장 불가, 오류 발생 : {error}")

if __name__ == "__main__":
    fetchRoutes()