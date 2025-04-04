# source/yolo_handler.py

import cv2
from ultralytics import YOLO
from config import YOLO_MODEL_PATH

# 모델 초기화
model = YOLO(YOLO_MODEL_PATH)

def detect_boxes(image):
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    return boxes