# backend/source/ai/generateParquet.py

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)
from source.utils.getDayType import getDayType

def main():
    # 경로 설정
    ETA_TABLE_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed', 'eta_table')
    REALTIME_BUS_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'dynamicInfo', 'realtime_bus')
    WEATHER_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'dynamicInfo', 'weather')
    STDID_TO_STOPS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'stdid_to_stops.json')
    STDID_NUMBER_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'stdid_number.json')
    NX_NY_STOPS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'nx_ny_stops.json')

    # 날짜 설정 (엊그제 baseline, 어제 raw)
    # today = datetime.now()
    today = datetime(2025, 4, 25)  # 임시로 4월 25일로 고정
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)
    yesterday_str = yesterday.strftime('%Y%m%d')
    day_before_str = day_before.strftime('%Y%m%d')

    # 파일 경로
    baseline_path = os.path.join(ETA_TABLE_DIR, f'{day_before_str}.json')
    parquet_save_path = os.path.join(ETA_TABLE_DIR, f'{yesterday_str}.parquet')

    # 데이터 로드
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)

    with open(STDID_TO_STOPS_PATH, 'r') as f:
        stdid_to_stops = json.load(f)

    with open(STDID_NUMBER_PATH, 'r') as f:
        stdid_number = json.load(f)

    with open(NX_NY_STOPS_PATH, 'r') as f:
        nx_ny_stops = json.load(f)

    # 학습 데이터 리스트
    data_list = []

    # 라벨 인코더 준비
    route_encoder = LabelEncoder()
    node_encoder = LabelEncoder()

    route_encoder.fit(list(stdid_number.values()))
    node_encoder.fit(list(set(stdid_to_stops.values())))

    # 전체 STDID 탐색
    for stdid_folder in os.listdir(os.path.join(REALTIME_BUS_DIR)):
        stdid_path = os.path.join(REALTIME_BUS_DIR, stdid_folder)
        if not os.path.isdir(stdid_path):
            continue

        for file in os.listdir(stdid_path):
            if not file.startswith(yesterday_str):
                continue

            hhmm = file.split('_')[-1].split('.')[0]
            stdid_hhmm = f'{stdid_folder}_{hhmm}'

            realtime_file_path = os.path.join(stdid_path, file)
            with open(realtime_file_path, 'r') as f:
                realtime_data = json.load(f)

            stop_reached_logs = realtime_data.get('stop_reached_logs', [])

            for log in stop_reached_logs:
                ord_num = str(log['ord'])
                arrival_time_str = log['time']
                if stdid_hhmm not in baseline_data or ord_num not in baseline_data[stdid_hhmm]:
                    continue

                baseline_time_str = baseline_data[stdid_hhmm][ord_num]

                # 시간 변환
                try:
                    baseline_time = datetime.strptime(baseline_time_str, '%H:%M:%S')
                    arrival_time = datetime.strptime(arrival_time_str, '%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    continue

                baseline_elapsed = (baseline_time.hour * 3600 + baseline_time.minute * 60 + baseline_time.second)
                actual_elapsed = (arrival_time.hour * 3600 + arrival_time.minute * 60 + arrival_time.second)
                delta_elapsed = actual_elapsed - baseline_elapsed

                # weather 매핑
                key_for_weather = f"{stdid_folder}_{ord_num}"
                nx_ny_key = nx_ny_stops.get(key_for_weather)
                if not nx_ny_key:
                    continue

                weather_file_path = os.path.join(WEATHER_DIR, f'{yesterday_str}_0000.json')
                with open(weather_file_path, 'r') as f:
                    weather_data = json.load(f)

                weather = weather_data.get(nx_ny_key, {'PTY': 0, 'RN1': 0, 'T1H': 20})

                # 요일 처리
                day_type = getDayType(yesterday)
                if day_type == 'weekday':
                    weekday_label = 0
                elif day_type == 'saturday':
                    weekday_label = 1
                else:
                    weekday_label = 2

                # 출발시간 처리 (sin/cos 변환)
                departure_time = datetime.strptime(hhmm, '%H%M')
                departure_seconds = departure_time.hour * 3600 + departure_time.minute * 60
                departure_time_sin = np.sin(2 * np.pi * departure_seconds / 86400)
                departure_time_cos = np.cos(2 * np.pi * departure_seconds / 86400)

                route_name = stdid_number.get(stdid_folder)
                node_id = stdid_to_stops.get(f'{stdid_folder}_{ord_num}')
                if route_name is None or node_id is None:
                    continue

                data_list.append({
                    'route_id_encoded': route_encoder.transform([route_name])[0],
                    'node_id_encoded': node_encoder.transform([node_id])[0],
                    'stop_ord': int(ord_num),
                    'departure_time_sin': departure_time_sin,
                    'departure_time_cos': departure_time_cos,
                    'weekday': weekday_label,
                    'PTY': weather['PTY'],
                    'RN1': weather['RN1'],
                    'T1H': weather['T1H'],
                    'baseline_elapsed': baseline_elapsed,
                    'actual_elapsed': actual_elapsed,
                    'delta_elapsed': delta_elapsed,
                })

    # parquet 저장
    if data_list:
        df = pd.DataFrame(data_list)
        os.makedirs(os.path.dirname(parquet_save_path), exist_ok=True)
        df.to_parquet(parquet_save_path, index=False)
        print(f"Parquet 생성 완료: {parquet_save_path}")
    else:
        print("생성할 데이터가 없습니다.")

if __name__ == "__main__":
    main()