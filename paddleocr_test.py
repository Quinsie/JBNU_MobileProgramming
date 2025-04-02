from paddleocr import PaddleOCR

ocr = PaddleOCR()
result = ocr.ocr("AI/image/temp.jpg")
print(result)