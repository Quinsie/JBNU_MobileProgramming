# source/config.py

import os

# Set Root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 모델 경로
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolov8n.pt")

# 테스트 이미지 경로
TEST_IMAGE_PATH = os.path.join(ROOT_DIR, "datasets", "images", "train", "reciept0001.png")

# OCR 설정
OCR_LANG = "korean"
USE_ANGLE_CLS = True