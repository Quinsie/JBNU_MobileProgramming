# source/ocr_handler.py

from paddleocr import PaddleOCR
from config import OCR_LANG, USE_ANGLE_CLS

# OCR 초기화
ocr = PaddleOCR(use_angle_cls=USE_ANGLE_CLS, lang=OCR_LANG)

# OCR 가동 함수
def extract_text_from_image(image):
    results = ocr.ocr(image, cls=True)
    texts = []

    for line in results:
        for box_info in line:
            text, confidence = box_info[1]
            text = text.strip()
            texts.append(text)

    return texts
