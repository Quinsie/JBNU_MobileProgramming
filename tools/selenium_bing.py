from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import os
import sys
import hashlib
import urllib.request
from PIL import Image
from io import BytesIO

MIN_WIDTH, MIN_HEIGHT = 224, 224
SAVE_DIR = os.path.join("datasets", "raw", "bing")
KEYWORDS = ["영수증", "영수증 사진", "영수증 이미지", "계산서", "매장 영수증"]

os.makedirs(SAVE_DIR, exist_ok=True)

def get_image_hash(image_data):
    return hashlib.md5(image_data).hexdigest()

def scroll_to_bottom(driver):
    scroll_pause_time = 1.0
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_count = 0

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)
        scroll_count += 1

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    print(f"[INFO] 스크롤 {scroll_count}회 완료")

def get_existing_image_count():
    return len([f for f in os.listdir(SAVE_DIR) if f.lower().endswith((".jpg", ".png"))])

def crawl_bing_images(keyword, start_index):
    print(f"[INFO] '{keyword}' 검색 시작")
    options = Options()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)

    url = f"https://www.bing.com/images/search?q={keyword}&form=HDRSC2"
    driver.get(url)

    scroll_to_bottom(driver)

    thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
    print(f"[INFO] 탐색된 이미지 수: {len(thumbnails)}")

    count = start_index
    skipped = 0
    seen_hashes = set()

    for img in thumbnails:
        try:
            src = img.get_attribute("src") or img.get_attribute("data-src")
            if not src or not src.startswith("http"):
                skipped += 1
                continue

            req = urllib.request.Request(src, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                image_data = resp.read()

            image = Image.open(BytesIO(image_data))
            if image.width < MIN_WIDTH or image.height < MIN_HEIGHT:
                skipped += 1
                continue

            img_hash = get_image_hash(image_data)
            if img_hash in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(img_hash)

            filename = os.path.join(SAVE_DIR, f"{count:05}.jpg")
            with open(filename, 'wb') as f:
                f.write(image_data)
            print(f"[{count}] 저장 완료")
            count += 1

        except Exception as e:
            skipped += 1
            continue

    driver.quit()
    print(f"[DONE] 총 {count - start_index}장 저장 완료, {skipped}장 스킵됨")

if __name__ == "__main__":
    start_idx = get_existing_image_count()
    for keyword in KEYWORDS:
        crawl_bing_images(keyword, start_idx)
        start_idx = get_existing_image_count()
