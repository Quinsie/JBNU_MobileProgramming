import easyocr
import cv2
import numpy as np
import os
import re # 불필요한 기호/문자 후처리 필터링

def resize_and_pad(image, target_size = (1000, 1600), pad_color = 255):
    height, width = image.shape[:2]
    target_height = target_size[1]
    target_width = target_size[0]

    # 비율에 맞춰 resize
    scale = min(target_width / width, target_height / height)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized_image = cv2.resize(image, (resized_width, resized_height), interpolation = cv2.INTER_AREA)

    # 패딩 값 계산
    padding_left = (target_width - resized_width) // 2
    padding_right = target_width - resized_width - padding_left
    padding_top = (target_height - resized_height) // 2
    padding_bottom = target_height - resized_height - padding_top

    # 패딩 적용
    padded_image  = cv2.copyMakeBorder(resized_image, padding_top, padding_bottom, padding_left, padding_right, 
                                       borderType = cv2.BORDER_CONSTANT, value = pad_color)
    
    return padded_image


# easyocr 가동
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# 이미지 read
image_path = "AI/image/temp.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8)) # 조명 문제 개선 및 텍스트 대비 증가 > 인식률 향상상
processed_image = resize_and_pad(image)

# 파일 이름 자동 생성
file_dir, file_name = os.path.split(image_path)
file_base, file_ext = os.path.splitext(file_name)
output_path = os.path.join(file_dir, f"resized_{file_base}{file_ext}")
cv2.imwrite(output_path, processed_image)

# 이진화/노이즈 제거는 오히려 이미지를 손상시킬 수도 있으니 적당히 상황을 봐 가면서 생각해야할듯.
# 근데 영수증 사진이래봐야 대부분 영수증 사진을 올곧게 찍을 테니 큰 신경을 쓸 필요가 있을까?하는 의문은 든다.

# 적응형 이진화 (Adaptive Thresholding), 고정 임계값은 조명 차이에 민감하여 OCR과 같은 작업에 최적화
# 너무 강하게 적용하면 글자가 뭉칠 수도 있으니 blocksize와 C값을 조절하며 실험이 필요하다. (뒷쪽 숫자 두개)
#image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)

# 노이즈 제거 : 점 노이즈나 작은 얼룩이 있으면 인식률이 저하된다.
#image = cv2.fastNlMeansDenoising(image, h=30)

# decoder = beamsearch; 결과 품질은 향상되지만 속도는 저하됨.
results = reader.readtext(processed_image, decoder = 'beamsearch', contrast_ths = 0.05, adjust_contrast = 0.7, text_threshold = 0.6)

for result in results:
    clean_text = re.sub(r'[^가-힣a-zA-Z0-9\s.,:/\-]', '', result[1])
    print(clean_text)