import torch
import paddle
from paddleocr import PaddleOCR

# PyTorch & CUDA 체크
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

# Paddle & CUDA 체크
print("Paddle Compiled with CUDA:", paddle.fluid.core.is_compiled_with_cuda())

# PaddleOCR 실행 (한국어 지원)
ocr = PaddleOCR(lang='korean', use_angle_cls=True)

# GPU 사용 여부 확인
print("Paddle Device:", paddle.device.get_device())  # 'gpu:0'이면 정상

# OCR 수행 (이미지 경로 설정)
image_path = 'AI/image/temp.jpg'  # 테스트할 이미지 경로
result = ocr.ocr(image_path)

# 결과 출력
for line in result:
    print(line)