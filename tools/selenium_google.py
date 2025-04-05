# tools/selenium_google.py

import os
import sys
import time
import requests
import urllib.request
from PIL import Image
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

# constant
MIN_WIDTH = 100
MIN_HEIGHT = 100

def crawl_google_images_selenium(keyword, max_count = 100):
    # 이미지셋 저장 디렉토리 구성, 있으면 무시
    save_dir = f"datasets/raw/google/{keyword.replace(' ', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    # chromedriver 경로: 현재 스크립트 기준 상대경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    driver_path = os.path.join(current_dir, "chromedriver.exe")
    service = Service(executable_path=driver_path)  # Service 객체로 래핑

    # 브라우저 UI 없이 실행
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # chromedriver 명시적 경로 지정, Selenium Webdriver 초기화
    # service 객체를 통해 chromedriver 경로 설정
    driver = webdriver.Chrome(service = service, options = chrome_options)

    # 구글 검색 페이지로 이동.
    print(f"[INFO] '{keyword}' 검색 시작")
    search_url = f"https://www.google.com/search?q={keyword}&tbm=isch"
    driver.get(search_url)
    time.sleep(2)

    # 스크롤을 반복하여 더 많은 이미지를 로드
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_num = 1000 # 스크롤 횟수.
    for _ in range(scroll_num):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # 이미지 태그를 선택 (구글 이미지 기준)
    images = driver.find_elements(By.CSS_SELECTOR, "img.rg_i")

    # 이미지 URL 추출 후 다운로드
    count = 0 # 현재까지 다운로드 성공한 개수수
    for img in images:
        try:
            src = img.get_attribute("src") or img.get_attribute("data-src") or img.get_attribute("data-lazy")
            if not src or not src.startswith("http"):
                continue  # 건너뜀

            if src: # http로 시작하면 외부 URL임. 광고 등 걸러냄.
                filename = os.path.join(save_dir, f"{count:04}.jpg") # 파일명 설정

                # PIL로 해상도 검사
                response = requests.get(src, timeout = 5)
                img = Image.open(BytesIO(response.content))
                if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
                    print(f"[SKIP] 해상도 낮음 : {img.width}×{img.height}")
                    continue

                urllib.request.urlretrieve(src, filename) # 저장
                count += 1
                print(f"[{count}] 저장됨: {filename}")
                if count >= max_count:
                    break
                
        except Exception as error: # 실패했을 때 로그 출력
            print(f"[ERROR] 이미지 저장 실패: {error}")

    # 전부 끝난 이후 드라이버 종료
    driver.quit()
    print(f"[DONE] 총 {count}장 저장 완료")

if __name__ == "__main__":
    # 인수 받아서 크롤링 함수로 넘김
    num = int(sys.argv[1])
    keywords = ["영수증", "영수증 사진", "영수증 이미지", "빌지", "계산서",
                "음식점 영수증", "편의점 영수증", "마트 영수증", "마트 계산서"]

    for keyword in keywords:
        crawl_google_images_selenium(keyword, num)