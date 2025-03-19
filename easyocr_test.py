import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['ko', 'en'], gpu=True)
image = cv2.imread("AI/image/temp.jpg", cv2.IMREAD_GRAYSCALE)

# 이진화/노이즈 제거는 오히려 이미지를 손상시킬 수도 있으니 적당히 상황을 봐 가면서 생각해야할듯.
# 근데 영수증 사진이래봐야 대부분 영수증 사진을 올곧게 찍을 테니 큰 신경을 쓸 필요가 있을까?하는 의문은 든다.

# 적응형 이진화 (Adaptive Thresholding), 고정 임계값은 조명 차이에 민감하여 OCR과 같은 작업에 최적화
# 너무 강하게 적용하면 글자가 뭉칠 수도 있으니 blocksize와 C값을 조절하며 실험이 필요하다. (뒷쪽 숫자 두개)
#image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

# 노이즈 제거 : 점 노이즈나 작은 얼룩이 있으면 인식률이 저하된다.
#image = cv2.fastNlMeansDenoising(image, h=30)

results = reader.readtext(image)

for result in results:
    print(result[1])