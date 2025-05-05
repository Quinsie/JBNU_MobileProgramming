# backend/source/ai/generateFirstETATable2.py

import os
import sys
import json
import torch
import time
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.getDayType import getDayType

# 날짜 설정
# TODAY = datetime.now()
TODAY = datetime(2025, 4, 26)  # 지금 날짜
YESTERDAY_DATE = TODAY - timedelta(days=1)  # 학습용 어제 날짜
TARGET_DATE = TODAY  # 추론 목표 날짜

YESTERDAY_STR = YESTERDAY_DATE.strftime("%Y%m%d")
TARGET_STR = TARGET_DATE.strftime("%Y%m%d")

PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", f"{YESTERDAY_STR}.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model", f"{YESTERDAY_STR}.pth")
BASELINE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{(YESTERDAY_DATE - timedelta(days=1)).strftime('%Y%m%d')}.json")
SAVE_JSON_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY_STR}.json")
REALTIME_BUS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")
LOOKUP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_lookup.json")

STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
with open(STDID_NUMBER_PATH, 'r') as f:
    stdid_number = json.load(f)

with open(LOOKUP_PATH, 'r') as f:
    nx_ny_lookup = json.load(f)

INPUT_DIM = 6
EMBEDDING_DIMS = {
    'route_id': (500, 8),
    'node_id': (3200, 16),
    'weekday_timegroup': (24, 4),  # ✅ 수정
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ETA_MLP(torch.nn.Module):
    def __init__(self):
        super(ETA_MLP, self).__init__()
        self.route_emb = torch.nn.Embedding(*EMBEDDING_DIMS['route_id'])
        self.node_emb = torch.nn.Embedding(*EMBEDDING_DIMS['node_id'])
        self.weekday_timegroup_emb = torch.nn.Embedding(*EMBEDDING_DIMS['weekday_timegroup'])  # ✅ 수정

        self.fc1 = torch.nn.Linear(INPUT_DIM + 8 + 16 + 4, 128)  # ✅ 수정
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, route_id, node_id, weekday_timegroup, dense_feats):  # ✅ 수정
        route_emb = self.route_emb(route_id)
        node_emb = self.node_emb(node_id)
        weekday_emb = self.weekday_timegroup_emb(weekday_timegroup)  # ✅ 수정
        x = torch.cat([dense_feats, route_emb, node_emb, weekday_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

def load_forecast_candidates(base_today):
    candidates = []
    for offset in range(3):
        file_date = (base_today - timedelta(days=offset)).strftime('%Y%m%d')
        path = os.path.join(FORECAST_DIR, f"{file_date}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                candidates.append({"date": file_date, "data": data})
    return candidates

def get_forecast_values(forecasts_list, forecast_timestamp, nx_ny):
    def is_valid(val):
        return val not in [None, {}, "null"]

    def extract_hour(ts):  # "20250509_1400" → 14
        return int(ts[-4:-2])

    # 1. 타임스탬프 fallback시 사용할 후보들 (3시간 전까지)
    target_hour = extract_hour(forecast_timestamp)
    timestamp_candidates = [
        forecast_timestamp[:-4] + f"{hour:02d}00"
        for hour in range(target_hour, target_hour - 4, -1)
        if hour >= 0
    ]

    # 2. 예보 파일 fallback: today → today-1 → today-2
    for forecast in forecasts_list:
        forecast_data = forecast["data"]

        for ts in timestamp_candidates:
            if ts not in forecast_data:
                continue

            # 3. 정확한 grid 먼저
            if is_valid(forecast_data[ts].get(nx_ny)):
                return forecast_data[ts][nx_ny]

            # 4. 주변 grid 탐색 (±1)
            try:
                x, y = map(int, nx_ny.split("_"))
            except:
                continue

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    alt_key = f"{x + dx}_{y + dy}"
                    if is_valid(forecast_data[ts].get(alt_key)):
                        return forecast_data[ts][alt_key]

    # 5. 다 실패 → 기본값
    return {'PTY': 0, 'PCP': 0.0, 'TMP': 20.0}

def process_std_folder(stdid_folder_args):
    stdid_folder, REALTIME_BUS_DIR, YESTERDAY_STR = stdid_folder_args
    folder_path = os.path.join(REALTIME_BUS_DIR, stdid_folder)
    recovered = {}

    if not os.path.isdir(folder_path):
        return recovered

    for file in os.listdir(folder_path):
        if not file.startswith(YESTERDAY_STR):
            continue
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for log in data.get('stop_reached_logs', []):
                ord_num = str(log['ord'])
                time_str = log['time'][-8:]
                recovered.setdefault(f"{stdid_folder}_{file.split('_')[-1].split('.')[0]}", {})[ord_num] = time_str
        except:
            continue
    return recovered

def postprocess_eta_table(eta_table, baseline_table, realtime_raw_dir):
    stdid_folders = os.listdir(realtime_raw_dir)
    with Pool(cpu_count()) as pool:
        results = pool.map(process_std_folder, [(stdid, REALTIME_BUS_DIR, YESTERDAY_STR) for stdid in stdid_folders])

        # 1. baseline 보완 (빈 경우에만)
    for stdid_hhmm, stops in baseline_table.items():
        if stdid_hhmm not in eta_table:
            eta_table[stdid_hhmm] = {}
        for ord_str, time_val in stops.items():
            if ord_str not in eta_table[stdid_hhmm]:
                eta_table[stdid_hhmm][ord_str] = time_val

    # 2. raw도 보완 (1, 2 다 없을 때만)
    for partial_table in results:
        for stdid_hhmm, stops in partial_table.items():
            if stdid_hhmm not in eta_table:
                eta_table[stdid_hhmm] = {}
            for ord_str, time_val in stops.items():
                if ord_str not in eta_table[stdid_hhmm]:
                    eta_table[stdid_hhmm][ord_str] = time_val

    return eta_table

def main():
    start_time = time.time()
    print(f"ETA Table 생성 시작... (target={TARGET_STR})")

    df = pd.read_parquet(PARQUET_PATH)
    forecasts_list = load_forecast_candidates(TODAY)

    weekday_timegroups = [f"{d}_{t}" for d in ['weekday', 'saturday', 'holiday'] for t in range(8)]

    weekday_timegroup_encoder = LabelEncoder()
    route_encoder = LabelEncoder()
    node_encoder = LabelEncoder()

    weekday_timegroup_encoder.fit(weekday_timegroups)
    route_encoder.fit(list(stdid_number.values()))
    node_encoder.fit(df['node_id'].unique())

    # 요일 인코딩
    day_type = getDayType(TARGET_DATE)

    weekday_timegroup_list = []
    for idx, row in df.iterrows():
        departure_group = int(row['departure_time_group'])
        combo = f"{day_type}_{departure_group}"
        encoded = weekday_timegroup_encoder.transform([combo])[0]
        weekday_timegroup_list.append(encoded)

    weekday_timegroup_tensor = torch.tensor(weekday_timegroup_list, dtype=torch.long).to(DEVICE)

    X_dense_rows = []
    route_id_encoded_list = []
    node_id_encoded_list = []

    for idx, row in df.iterrows():
        departure_hhmm = int(row['departure_hhmm'])
        dep_hour = departure_hhmm // 100
        dep_min = departure_hhmm % 100
        forecast_timestamp = f"{TARGET_DATE.strftime('%Y%m%d')}_{dep_hour:02d}00"

        stdid = [k for k, v in stdid_number.items() if v == row['route_id']]
        stop_ord = str(int(row['stop_ord']))
        nx_ny = nx_ny_lookup.get(f"{stdid[0]}_{stop_ord}") if stdid else None

        forecast = get_forecast_values(forecasts_list, forecast_timestamp, nx_ny)

        X_dense_rows.append([
            row['departure_time_sin'],
            row['departure_time_cos'],
            row['departure_time_group'],
            forecast['PTY'],
            forecast['PCP'],
            forecast['TMP']
        ])

        route_id_encoded_list.append(route_encoder.transform([row['route_id']])[0])
        node_id_encoded_list.append(node_encoder.transform([row['node_id']])[0])

    X_dense = torch.tensor(X_dense_rows, dtype=torch.float32).to(DEVICE)
    route_id_tensor = torch.tensor(route_id_encoded_list, dtype=torch.long).to(DEVICE)
    node_id_tensor = torch.tensor(node_id_encoded_list, dtype=torch.long).to(DEVICE)

    model = ETA_MLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred_delta = model(route_id_tensor, node_id_tensor, weekday_timegroup_tensor, X_dense).cpu().numpy()

    with open(BASELINE_PATH, 'r') as f:
        baseline_table = json.load(f)

    eta_table = {}
    for idx, row in df.iterrows():
        route_id = row['route_id']
        stop_ord = str(int(row['stop_ord']))
        departure_hhmm = int(row['departure_hhmm'])
        stdid_candidates = [stdid for stdid, rname in stdid_number.items() if rname == route_id]
        if not stdid_candidates:
            continue
        stdid = stdid_candidates[0]
        stdid_hhmm = f"{stdid}_{departure_hhmm:04d}"

        dep_hour = departure_hhmm // 100
        dep_min = departure_hhmm % 100
        dep_time = datetime(TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day, dep_hour, dep_min, 0)

        dep_seconds = dep_hour * 3600 + dep_min * 60
        baseline_elapsed = row['baseline_elapsed'] - dep_seconds
        if baseline_elapsed < 0:
            baseline_elapsed = 0

        delta = pred_delta[idx]
        final_elapsed = baseline_elapsed + delta
        if final_elapsed < 0:
            final_elapsed = 0

        eta_time = dep_time + timedelta(seconds=int(final_elapsed))
        eta_time_str = eta_time.strftime("%H:%M:%S")

        if stdid_hhmm not in eta_table:
            eta_table[stdid_hhmm] = {}
        eta_table[stdid_hhmm][stop_ord] = eta_time_str

    eta_table = postprocess_eta_table(eta_table, baseline_table, REALTIME_BUS_DIR, YESTERDAY_STR)

    os.makedirs(os.path.dirname(SAVE_JSON_PATH), exist_ok=True)
    with open(SAVE_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(eta_table, f, indent=2, ensure_ascii=False)

    print(f"ETA Table 생성 완료: {SAVE_JSON_PATH}")
    print("총 소요시간: ", time.time() - start_time, "sec")

if __name__ == "__main__":
    main()