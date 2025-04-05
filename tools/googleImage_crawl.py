# tools/googleImage_crawl.py

import sys
from icrawler.builtin import GoogleImageCrawler

if __name__ == "__main__":
    # 하단 키워드를 바탕으로 한 크롤링 예정
    keywords = ["영수증", "편의점 영수증", "편의점 영수증 사진", "마트 영수증", "마트 영수증 사진"]
    # 크롤링 시 라이센스 구성
    licenses = ["noncommercial", "commercial", "noncommercial,modify", "commercial,modify"]

    # 프로그램 실행 시 인수 전달로 크롤링 숫자 지정
    num = int(sys.argv[1]) if len(sys.argv[1]) > 1 else 0

    for i in range(4):
        # 라이센스 설정
        license_type = licenses[i]

        # 상단 작성한 키워드에 대해 반복하며 진행 상황을 출력
        for i in range(len(keywords)):
            keyword = keywords[i]
            print(f"[INFO] Now crawling: {keyword}")

            # 크롤러 저장 경로 설정
            google_crawler = GoogleImageCrawler(storage = {'root_dir': f'datasets/raw/{license_type}/{keyword.replace(" ", "_")}'})
            google_crawler.crawl(
                keyword = keyword,
                max_num = num,
                filters = {
                    'type': 'photo',
                    'license': license_type
                }
            )