# README.md
- 본 프로그램에 대한 안내문서입니다.
---
### 목차
 1. 프로젝트 디렉터리 구조
 2. 
---
### 1. 📁 프로젝트 디렉터리 구조
```text
AI/
├── datasets/
│   ├── images/
│   │   ├── train/            # 학습용 이미지
│   │   └── val/              # 검증용 이미지
│   └── labels/
│       ├── train/            # 학습용 이미지 정답
│       └── val/              # 검증용 이미지 정답
├── models/
│   └── yolov8n.pt            # 모델 파일 (추후 best.pt로 대체 예정)
├── source/
│   ├── config.py             # 경로 등 설정 파일
│   ├── food_manager.py       # 메인 실행 파일
│   ├── ocr_handler.py        # PaddleOCR 관련 모듈
│   ├── yolo_handler.py       # YOLO 관련 모듈
│   └── utils.py              # 보조 유틸 함수
├── PROJECT_PLAN.md           # 프로젝트 계획 문서
└── README.md                 # 프로젝트 설명 문서
```
---
