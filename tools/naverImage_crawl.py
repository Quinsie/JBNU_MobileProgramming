# tools/naverImage_crawl.py

import os
import sys
import requests # 웹페이지 요청 HTTP 클라이언트
from bs4 import BeautifulSoup # 파싱
import urllib.parse # URL 인코딩

def crawl_naver_images(keyword: str, max_images: int = 100):
    headers = {'User-Agent': 'Mozilla/5.0'}

    # 키워드 별 폴더 분리, 이미 폴더 있으면 pass
    save_dir = f'datasets/raw/naver/{keyword.replace(" ", "_")}'
    os.makedirs(save_dir, exist_ok=True)

    # 검색어를 URL 인코딩
    encoded = urllib.parse.quote(keyword)
    url = f'https://search.naver.com/search.naver?where=image&sm=tab_jum&query={encoded}'

    # URL 요청 후 HTML을 가져와서 BeautifulSoup으로 파싱
    print(f'[INFO] 검색 키워드: {keyword}')
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')

    # HTML 내 모든 <img> 태그를 가져온다.
    images = soup.find_all('img')

    count = 0
    for i in range(len(images)):
        img = images[i]
        # 이미지 URL을 가져온다. <src>또는 <data-src> 속성 이용용
        img_url = img.get('src') or img.get('data-src')

        # 유효한 이미지 URL이라면:
        if img_url and img_url.startswith('http'):
            try: # 다운로드
                img_data = requests.get(img_url, headers=headers).content
                with open(os.path.join(save_dir, f'{count:04}.jpg'), 'wb') as f:
                    f.write(img_data)
                count += 1
                if count >= max_images:
                    break
            # 아닐 때 예외처리
            except Exception as e:
                print(f'[ERROR] 다운로드 실패: {img_url} - {e}')
                continue

    print(f'[DONE] {count}장 저장됨: {save_dir}')

if __name__ == "__main__":
    # 예외처리
    if len(sys.argv) < 2:
        print("사용법: python tools/naverImage_crawl.py <저장할 이미지 수>")
        sys.exit(1)

    keywords = ["영수증"]
    # 인수 입력받아 함수 실행
    num = int(sys.argv[1])
    for i in range(len(keywords)):
        keyword = keywords[i]
        crawl_naver_images(keyword, num)