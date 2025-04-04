import torch
import paddle
import cv2
from paddleocr import PaddleOCR
from paddleocr import draw_ocr
from PIL import Image

# PaddleOCR 실행 (한국어 지원)
ocr = PaddleOCR(lang='korean', use_angle_cls=True)

# OCR 수행 (이미지 경로 설정)
image_path = 'AI/image/reciept001.png'  # 테스트할 이미지 경로
result = ocr.ocr(image_path)

# 결과 출력
for line in result:
    for box, (text, confidence) in line:
        print(f"Text: {text}, Confidence: {confidence:.2f}")

# 인식 결과 시각화
image = Image.open(image_path).convert('RGB')
boxes = [elements[0] for elements in result[0]]
txts = [elements[1][0] for elements in result[0]]
scores = [elements[1][1] for elements in result[0]]

# 폰트 설정, 오류 배제
font_path = "C:/Windows/Fonts/malgun.ttf"
annotated = draw_ocr(image, boxes, txts, scores, font_path = font_path)
cv2.imwrite('AI/image/modified/reciept001_visualized.jpg', annotated)

# 인식 결과 텍스트 추출
with open("AI/text/reciept001_text.txt", "w", encoding="utf-8") as f:
    for line in result:
        for _, (text, conf) in line:
            f.write(f"{text} ({conf:.2f})\n")