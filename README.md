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
├── android/
├── backend/
│   ├── source/
│   │   ├── scripts/
│   │   │   ├── buildStopIndex.py
│   │   │   ├── departureCacheGenerator.py
│   │   │   ├── fetchDepartureTimetables.py
│   │   │   ├── fetchRouteId.py
│   │   │   ├── fetchStops.py
│   │   │   ├── fetchSubList.py
│   │   │   ├── fetchTrafficVtxList.py
│   │   │   ├── fetchVTX.py
│   │   │   ├── mapVtxToRoadId.py
│   │   │   ├── runAll.py
│   │   │   ├── scheduler.py
│   │   │   ├── trackSingleBus.py
│   │   │   ├── trafficCollector.py
│   │   │   └── weatherCollecter.py
│   │   └── utils/
│   │       ├── convertToGrid.py
│   │       ├── getDayType.py
│   │       ├── haversine.py
│   │       └── logger.py
│   └── data/
│       ├── raw/
│       │   ├── staticInfo/
│       │   │   ├── departure_timetables/
│       │   │   ├── stops/
│       │   │   ├── subList/
│       │   │   ├── vtx/
│       │   │   ├── routes.json
│       │   │   └── traf_vtxlist.json
│       │   └── dynamicInfo/
│       │       ├── realtime_bus/
│       │       ├── traffic/
│       │       └── weather/
│       └── processed/
│           ├── departure_cache/
│           ├── departure_timetables/
│           ├── stop_to_routes/
│           ├── vtx_mapped/
│           ├── grouped_weather.json
│           └── nx_ny_coords.json
├── PROJECT_PLAN.md
└── README.md
```
---
### 2. 항목별 설명