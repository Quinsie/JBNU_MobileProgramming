# source/food_manager.py

import cv2
from config import TEST_IMAGE_PATH
from yolo_handler import detect_boxes
from ocr_handler import extract_text_from_image
from utils import filter_product_names

def main():
    # 이미지 로드
    image = cv2.imread(TEST_IMAGE_PATH)

    # 박스 감지
    boxes = detect_boxes(image)

    product_names = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]

        texts = extract_text_from_image(cropped)
        product_names.extend(texts)

    # 필터링 및 중복 제거
    filtered_names = filter_product_names(product_names)

    print("추출된 상품명 후보:")
    for name in filtered_names:
        print("-", name)

if __name__ == "__main__":
    main()