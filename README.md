# README.md
- 본 프로그램에 대한 안내문서입니다.
---
### 목차
 1. 프로젝트 디렉터리 구조
 2. 항목 별 설명
---
### 1. 📁 프로젝트 디렉터리 구조
```text
AI/
├── android/                  # Android Studio 프로젝트 (앱 UI/UX, 카메라, 피드백 등)
│   └── ReceiptApp/           # 실제 앱 디렉토리
│       └── ...                     
│
├── backend/                  # 백엔드 분석/추론 엔진 (YOLO, OCR, 알림 등)
│   ├── models/
│   │   └── yolov8n.pt        # YOLOv8 모델 (→ 학습 후 best.pt로 교체 예정)
│   ├── config.py             # 공통 설정 (경로, 해상도 등)
│   ├── food_manager.py       # 전체 파이프라인 메인 실행
│   ├── ocr_handler.py        # PaddleOCR 처리
│   ├── yolo_handler.py       # YOLO 감지 처리
│   ├── utils.py              # 해시 비교, 해상도 필터링 등 보조 기능
│   └── database/
│       └── food_db.json      # 음식별 소비기한 캐시 (로컬 기반)
│
├── datasets/                 # 데이터 저장소 (학습 + 크롤링)
│   ├── raw/                  # 크롤링 원본 이미지 (모든 검색엔진 통합)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── cache/                # OCR/YOLO 라벨 자동 생성 임시저장소 (필요시)
│
├── tools/                    # 데이터 수집 도구들
│   ├── selenium_bing.py
│   ├── selenium_naver.py
│   ├── selenium_all_run.py   # 여러 검색엔진 병렬 실행 스크립트
│   └── label_assist.py       # 라벨링 도우미 (OCR 결과 포함된 bounding box 시각화 등)
│
├── README.md                 # 전체 프로젝트 개요
└── PROJECT_PLAN.md           # 프로젝트 진행 계획서

```
---
### 2. 항목 별 설명
#### food_manager.py
##### 메인 스크립트
 - 