# README.md
- 본 프로그램에 대한 안내문서입니다.
---
### 목차
 1. 프로젝트 디렉터리 구조
 2. 항목별 설명
---
### 1. 📁 프로젝트 디렉터리 구조
```text
BIS_APP
├── android/                        # 안드로이드 프론트엔드 디렉터리
├── backend/                        # AI/데이터 백엔드 디렉터리
│   ├── source/                         # 소스 코드 디렉터리
│   │   ├── scripts/                        # 실행 스크립트
│   │   │   ├── buildStopIndex.py               # 정류장 기준 노선 데이터 생성 : processed/stop_to_routes/
│   │   │   ├── departureCacheGenerator.py      # 버스 시간표를 요일별 캐싱 : processed/departure_cache
│   │   │   ├── fetchDepartureTimetables.py     # [API] 노선별 시간표 수집 : raw/staticInfo/departure_timetables
│   │   │   ├── fetchRouteId.py                 # [API] 전체 노선번호 수집 : raw/staticInfo/routes.json
│   │   │   ├── fetchStops.py                   # [API] 노선별 정차 정류장 수집 : raw/staticInfo/stops/
│   │   │   ├── fetchSubList.py                 # [API] 노선별 본선/분선 정보 수집 : raw/staticInfo/subList/
│   │   │   ├── fetchTrafficVtxList.py          # [API] 도로 위 특정 위치정보 수집 : raw/staticInfo/traf_vtxlist.json
│   │   │   ├── fetchVTX.py                     # [API] 노선별 경로 지리 정보 수집 : raw/staticInfo/vtx/
│   │   │   ├── mapVtxToRoadId.py               # 노선 경로를 도로 위치에 매핑 : processed/vtx_mapped/
│   │   │   ├── runAll.py                       # [SYSTEM] 모든 수집기를 특정 시간 간격에 맞춰 실행
│   │   │   ├── scheduler.py                    # [SYSTEM] 요일별 매핑된 버스 시간표 기준 실시간 버스정보 수집 트리거
│   │   │   ├── trackSingleBus.py               # [API] 실시간 버스정보 수집기 : raw/dynamicInfo/realtime_bus
│   │   │   ├── trafficCollector.py             # [API] 도로 위 특정 위치에 대한 실시간 혼잡도 수집 : raw/dynamicInfo/traffic/
│   │   │   └── weatherCollecter.py             # [API] 실시간 날씨정보 수집기 : raw/dynamicInfo/weather
│   │   └── utils/                          # 유틸리티티 스크립트
│   │       ├── convertToGrid.py                # 위경도 > 기상청 격자 좌표 편환기
│   │       ├── getDayType.py                   # 대한민국 공휴일 정보 수신, 평일/토요일/일요일+공휴일 판별기
│   │       ├── haversine.py                    # lat/long 기준 정확한 거리 측정 툴
│   │       └── logger.py                       # [SYSTEM] 로거 - 디버거
│   └── data/                       # 데이터 디렉터리
│       ├── raw/                        # 미가공 데이터
│       │   ├── staticInfo/                 # 정적 데이터
│       │   │   ├── departure_timetables/       # {노선 STDID}.json : 요일별 출발시간 데이터
│       │   │   ├── stops/                      # {노선 STDID}.json : 노선별 경로상 모든 정류장 데이터
│       │   │   ├── subList/                    # {노선번호}.json : 노선별 모든 본선/분선 정보 데이터
│       │   │   ├── vtx/                        # {노선 STDID}.json : 노선별 경로상 특정 지점 지리적 정보 데이터
│       │   │   ├── routes.json                 # 모든 노선번호가 기록된 데이터
│       │   │   └── traf_vtxlist.json           # 모든 도로 위 특정 위치정보 데이터
│       │   └── dynamicInfo/                # 동적 데이터
│       │       ├── realtime_bus/               # {노선 STDID}/{날짜_배차시간}.json : 실시간 추적된 버스 위치/도착시간 데이터
│       │       ├── traffic/                    # {날짜_시간}.json : 실시간 추출된 도로교통정보 데이터
│       │       └── weather/                    # {날짜_시간}.json : 30분 단위 실시간 추출된 날씨 데이터 (강수형태/강수량/기온)
│       └── processed/                  # 1차 가공 데이터
│           ├── departure_cache/            # 요일 종류별 버스 출발시간 리스트와 각 시간별로 매핑된 노선 데이터
│           ├── stop_to_routes/             # {정류장 ID}.json : 정류장 기준 지나가는 노선 데이터
│           ├── vtx_mapped/                 # {노선 STDID}.json : 노선별 경로상 특정 지점을 도로교통정보 지점에 매핑한 데이터
│           ├── grouped_weather.json        # 날씨 측정 지점이 똑같은 집합을 구분 및 매핑한 데이터
│           └── nx_ny_coords.json           # 날씨 측정 지점의 x, y 소분구를 LAT, LONG으로 매핑한 데이터
├── .gitignore              # 깃허브 풀/푸시 필터링
├── PROJECT_PLAN.md         # 프로젝트 계획문서
└── README.md               # 프로젝트 구성 소개 문서
```
---
### 2. 항목별 설명