# backend/source/ai/generateParquet2.py
# combo version

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from functools import partial

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.getDayType import getDayType
from source.utils.logger import log

# departure_time 기준 시간대 그룹핑
def get_time_group(departure_time):
    minutes = departure_time.hour * 60 + departure_time.minute
    if 330 <= minutes < 420:     # 05:30~07:00
        return 0
    elif 420 <= minutes < 540:   # 07:00~09:00
        return 1
    elif 540 <= minutes < 720:   # 09:00~11:30
        return 2
    elif 690 <= minutes < 840:   # 11:30~14:00
        return 3
    elif 840 <= minutes < 1020:  # 14:00~17:00
        return 4
    elif 1020 <= minutes < 1140: # 17:00~19:00
        return 5
    elif 1140 <= minutes < 1260: # 19:00~21:00
        return 6
    else:                        # 21:00~00:00
        return 7

# multiprocessing용 함수
def process_std_folder(stdid_folder, args):
    (REALTIME_BUS_DIR, baseline_data, stdid_to_stops, stdid_number, nx_ny_stops,
     weather_files, yesterday_str, weekday_label, weekday_encoder,
     route_encoder, node_encoder) = args

    stdid_path = os.path.join(REALTIME_BUS_DIR, stdid_folder)
    if not os.path.isdir(stdid_path):
        return []

    data_list = []

    for file in os.listdir(stdid_path):
        if not file.startswith(yesterday_str):
            continue

        hhmm = file.split('_')[-1].split('.')[0]
        stdid_hhmm = f'{stdid_folder}_{hhmm}'

        realtime_file_path = os.path.join(stdid_path, file)
        with open(realtime_file_path, 'r') as f:
            realtime_data = json.load(f)

        stop_reached_logs = realtime_data.get('stop_reached_logs', [])

        for log_entry in stop_reached_logs:
            ord_num = str(log_entry['ord'])
            arrival_time_str = log_entry['time']
            if stdid_hhmm not in baseline_data or ord_num not in baseline_data[stdid_hhmm]:
                continue

            baseline_time_str = baseline_data[stdid_hhmm][ord_num]

            try:
                baseline_time = datetime.strptime(baseline_time_str, '%H:%M:%S')
                arrival_time = datetime.strptime(arrival_time_str, '%Y-%m-%d %H:%M:%S')
            except Exception as e:
                continue

            baseline_elapsed = baseline_time.hour * 3600 + baseline_time.minute * 60 + baseline_time.second
            actual_elapsed = arrival_time.hour * 3600 + arrival_time.minute * 60 + arrival_time.second
            delta_elapsed = actual_elapsed - baseline_elapsed

            # 시간대 그룹 분류
            departure_time = datetime.strptime(hhmm, '%H%M')
            departure_time_group = get_time_group(departure_time)
            weekday_timegroup = f"{weekday_label}_{departure_time_group}"
            weekday_timegroup_encoded = weekday_encoder.transform([weekday_timegroup])[0]

            # weather 매칭
            arrival_hhmm = arrival_time.hour * 100 + arrival_time.minute
            closest_weather_file = find_closest_weather_file(weather_files, arrival_hhmm)

            if closest_weather_file:
                try:
                    with open(closest_weather_file, 'r') as f:
                        weather_data = json.load(f)
                except Exception as e:
                    log("generateParquet", f"날씨 파일 로드 실패: {closest_weather_file}, 에러: {e}")
                    weather_data = {}
            else:
                log("generateParquet", f"날씨 파일 없음: STDID {stdid_folder}, 시간 {arrival_time_str}")
                weather_data = {}

            key_for_weather = f"{stdid_folder}_{ord_num}"
            nx_ny_key = nx_ny_stops.get(key_for_weather)

            if not nx_ny_key:
                log("generateParquet", f"nx_ny_stops 매칭 실패: {key_for_weather}")
                weather = {'PTY': 0, 'RN1': 0, 'T1H': 20}
            else:
                weather_info = weather_data.get(nx_ny_key)
                if not weather_info:
                    log("generateParquet", f"weather_data 매칭 실패: {nx_ny_key}")
                    weather = {'PTY': 0, 'RN1': 0, 'T1H': 20}
                else:
                    weather = {
                        'PTY': weather_info.get('PTY', 0),
                        'RN1': weather_info.get('RN1', 0),
                        'T1H': weather_info.get('T1H', 20)
                    }

            # 출발시간 처리
            departure_seconds = departure_time.hour * 3600 + departure_time.minute * 60
            departure_time_sin = np.sin(2 * np.pi * departure_seconds / 86400)
            departure_time_cos = np.cos(2 * np.pi * departure_seconds / 86400)
            actual_elapsed_from_departure = actual_elapsed - departure_seconds
            if actual_elapsed_from_departure < 0:
                actual_elapsed_from_departure = 0  # 혹시 음수 나오면 0으로

            route_name = stdid_number.get(stdid_folder)
            node_id = stdid_to_stops.get(f'{stdid_folder}_{ord_num}')
            if route_name is None or node_id is None:
                continue

            data_list.append({
                'route_id_encoded': route_encoder.transform([route_name])[0],
                'node_id_encoded': node_encoder.transform([node_id])[0],
                'weekday_timegroup_encoded': weekday_timegroup_encoded,
                'stop_ord': int(ord_num),
                'departure_time_sin': departure_time_sin,
                'departure_time_cos': departure_time_cos,
                'departure_time_group': departure_time_group,
                'PTY': weather['PTY'],
                'RN1': weather['RN1'],
                'T1H': weather['T1H'],
                'baseline_elapsed': baseline_elapsed,
                'actual_elapsed': actual_elapsed,
                'delta_elapsed': delta_elapsed,
                'route_id': route_name,
                'node_id': node_id,
                'departure_hhmm': int(hhmm),
            })

    return data_list

# weather 관련 함수들
def load_weather_files(weather_dir, target_date):
    weather_files = {}
    for file in os.listdir(weather_dir):
        if file.startswith(target_date):
            time_part = int(file.split('_')[-1].split('.')[0])
            weather_files[time_part] = os.path.join(weather_dir, file)
    return dict(sorted(weather_files.items()))

def find_closest_weather_file(weather_files, target_time):
    available_times = [t for t in weather_files.keys() if t <= target_time]
    if not available_times:
        return None
    return weather_files[max(available_times)]

def main():
    ETA_TABLE_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed', 'eta_table')
    REALTIME_BUS_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'dynamicInfo', 'realtime_bus')
    WEATHER_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'dynamicInfo', 'weather')
    PARQUET_DIR = os.path.join(BASE_DIR, 'data', 'preprocessed', 'first_train')
    STDID_TO_STOPS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'stdid_to_stops.json')
    STDID_NUMBER_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'stdid_number.json')
    NX_NY_STOPS_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'nx_ny_stops.json')

    start_time = time.time()
    # today = datetime.now()
    today = datetime(2025, 4, 25)
    yesterday = today - timedelta(days=1)
    day_before = today - timedelta(days=2)
    yesterday_str = yesterday.strftime('%Y%m%d')
    day_before_str = day_before.strftime('%Y%m%d')

    baseline_path = os.path.join(ETA_TABLE_DIR, f'{day_before_str}.json')
    parquet_save_path = os.path.join(PARQUET_DIR, f'{yesterday_str}.parquet')

    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)

    with open(STDID_TO_STOPS_PATH, 'r') as f:
        stdid_to_stops = json.load(f)

    with open(STDID_NUMBER_PATH, 'r') as f:
        stdid_number = json.load(f)

    with open(NX_NY_STOPS_PATH, 'r') as f:
        nx_ny_stops = json.load(f)

    weather_files = load_weather_files(WEATHER_DIR, yesterday_str)
    weekday_timegroups = [f"{d}_{t}" for d in ['weekday', 'saturday', 'holiday'] for t in range(8)]

    route_encoder = LabelEncoder()
    node_encoder = LabelEncoder()
    weekday_encoder = LabelEncoder()
    
    route_encoder.fit(list(stdid_number.values()))
    node_encoder.fit(list(set(stdid_to_stops.values())))
    weekday_encoder.fit(weekday_timegroups)

    day_type = getDayType(yesterday)

    stdid_folders = os.listdir(REALTIME_BUS_DIR)

    args = (REALTIME_BUS_DIR, baseline_data, stdid_to_stops, stdid_number, nx_ny_stops,
        weather_files, yesterday_str, day_type, weekday_encoder,
        route_encoder, node_encoder)

    with Pool(cpu_count()) as pool:
        results = pool.map(partial(process_std_folder, args=args), stdid_folders)

    # 결과 합치기
    all_data = [item for sublist in results for item in sublist]

    if all_data:
        df = pd.DataFrame(all_data)
        os.makedirs(os.path.dirname(parquet_save_path), exist_ok=True)
        df.to_parquet(parquet_save_path, index=False)
        log("generateParquet", f"Parquet 생성 완료: {parquet_save_path}")
    else:
        log("generateParquet", "생성할 데이터가 없습니다.")

    print("소요 시간: ", time.time() - start_time, "sec")

if __name__ == "__main__":
    main()