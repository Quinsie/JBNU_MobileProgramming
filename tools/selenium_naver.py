# tools/selenium_naver.py

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

# 이미 저장된 이미지 수를 기준으로 다음 파일 인덱스를 반환하는 함수
def get_next_index(save_dir):
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    # 없으면 0부터 시작
    if not existing: return 0
    # 순차 탐색으로 끝까지 가기 때문에 어떻게 보면 비효율인가? 개선할 여지는 있음. 근데 중요하진 않을듯.
    numbers = [int(os.path.splitext(f)[0]) for f in existing if f[:4].isdigit()]
    return max(numbers) + 1 if numbers else 0

def crawl_naver_images_selenium(keyword, max_count = 100):
    # 이미지셋 저장 디렉토리 구성, 있으면 무시
    save_dir = f"datasets/raw/naver/{keyword.replace(' ', '_')}"
    os.makedirs(save_dir, exist_ok=True)
    start_index = get_next_index(save_dir)

    # chromedriver 경로: 현재 스크립트 기준 상대경로
    current_dir = os.path.dirname(os.path.abspath(__file__))
    driver_path = os.path.join(current_dir, "chromedriver.exe")
    service = Service(executable_path=driver_path)  # Service 객체로 래핑

    # 브라우저 UI 없이 실행
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    # headless 브라우저 탐지 시도 우회
    chrome_options.add_argument("user-agent=Mozilla/5.0")
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')

    # chromedriver 명시적 경로 지정, Selenium Webdriver 초기화
    # service 객체를 통해 chromedriver 경로 설정
    driver = webdriver.Chrome(service = service, options = chrome_options)

    # 네이버 검색 페이지로 이동.
    print(f"[INFO] '{keyword}' 검색 시작")
    search_url = f"https://search.naver.com/search.naver?where=image&query={keyword}"
    driver.get(search_url)
    time.sleep(2)

    # 더보기 버튼 클릭 반복
    try:
        while True:
            more_button = driver.find_element(By.CLASS_NAME, "more_btn")
            more_button.click()
            time.sleep(2)
    except:
        print("[INFO] 더보기 버튼 없음 또는 클릭 완료")

    # 스크롤을 반복하여 더 많은 이미지를 로드
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_num = 100 # 스크롤 횟수.
    for _ in range(scroll_num):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            print("[INFO] 더 이상 스크롤할 수 없습니다.")
            break
        last_height = new_height

    # 이미지 태그를 선택 (네이버 이미지 뷰어 기준)
    images = driver.find_elements(By.CSS_SELECTOR, "div.thumb img")

    # 이미지 URL 추출 후 다운로드
    count = 0
    for img in images:
        try:
            src = img.get_attribute("src") # 이미지의 실제 경로 추출

            if src and src.startswith("http") and "data:image" not in src: # http로 시작하면 외부 URL임. 광고 등 걸러냄.
                filename = os.path.join(save_dir, f"{count:04}.jpg") # 파일이름 설정
                
                with urllib.request.urlopen(src, timeout = 5) as response:
                    img = Image.open(BytesIO(response.read()))
                    # PIL로 해상도 검사
                    if img.width < MIN_WIDTH or img.height < MIN_HEIGHT:
                        print(f"[SKIP] 해상도 낮음 : {img.width}×{img.height}")
                        continue
                    
                    # filename으로로 이미지 저장.
                    img.save(filename)

                count += 1
                print(f"[{count}] 저장됨: {filename}")

                # 요청한 사진보다 많은 사진이 저장되었을 때
                if count >= max_count: break

        except Exception as error: # 실패했을 때 로그 출력
            print(f"[ERROR] 이미지 저장 실패: {error}")
            continue

    # 전부 끝난 이후 드라이버 종료
    driver.quit()
    print(f"[DONE] 총 {count}장 저장 완료")

if __name__ == "__main__":
    # 인수 받아서 크롤링 함수로 넘김
    num = int(sys.argv[1])
    keywords = ["영수증", "영수증 사진", "영수증 이미지", "빌지", "계산서",
                "음식점 영수증", "편의점 영수증", "마트 영수증", "마트 계산서"]

    for keyword in keywords:
        crawl_naver_images_selenium(keyword, num)