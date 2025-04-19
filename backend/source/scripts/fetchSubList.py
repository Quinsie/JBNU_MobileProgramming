# backend/souce/scripts/fetchSubList.py

import os
import json
import time
import requests

URL = "http://www.jeonjuits.go.kr/bis/selectBisRouteSubList.do" # 본선/분선 API
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "subList"))
ROUTE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "staticInfo", "routes.json"))

def fetchSubLists():
    with open(ROUTE_PATH, "r", encoding = "utf-8") as file:
        route_list = json.load(file)
    
    os.makedirs(PATH, exist_ok = True)

    for num in route_list:
        payload = { # 버스 번호 기준으로 분선을 파악하기 위해 num을 인자로 줌 (기존 저장 데이터 기준)
            "locale": "ko-kr",
            "brt_no": num
        }
        try:
            res = requests.post(URL, data = payload, timeout = 5)
            res.raise_for_status()
            result = res.json()

            save_path = os.path.join(PATH, f"{num}.json")
            with open(save_path, "w", encoding = "utf-8") as file:
                json.dump(result, file, ensure_ascii = False, indent = 2)
            
            print(f"{num}번 버스 저장 완료")
        
        except Exception as error:
            print("저장 불가, 오류 발생 {error}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    fetchSubLists()