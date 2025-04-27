# backend/source/ai/generateFirstETATable.py
# dimension 6&7버전 table을 만든다. Forecast true/false 유의할 것.

import os
import sys
import json
import torch
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# 날짜 설정
TODAY = datetime.now()
# TODAY = datetime(2025, 4, 26)  # 지금 날짜
YESTERDAY_DATE = TODAY - timedelta(days=1)  # 학습용 어제 날짜
TARGET_DATE = TODAY  # 추론 목표 날짜

YESTERDAY_STR = YESTERDAY_DATE.strftime("%Y%m%d")
TARGET_STR = TARGET_DATE.strftime("%Y%m%d")

PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", f"{YESTERDAY_STR}_2.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model", f"{YESTERDAY_STR}_2.pth")
BASELINE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{(YESTERDAY_DATE - timedelta(days=1)).strftime('%Y%m%d')}_2.json")
SAVE_JSON_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY_STR}_2.json")
REALTIME_BUS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")

STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
with open(STDID_NUMBER_PATH, 'r') as f:
    stdid_number = json.load(f)

INPUT_DIM = 7 # 6//7 조절.
EMBEDDING_DIMS = {
    'route_id': (500, 8),
    'node_id': (3200, 16),
    'weekday': (3, 2),
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_FORECAST = False  # True면 예보 사용, False면 기존 방식, 4/28부터 True

class ETA_MLP(torch.nn.Module):
    def __init__(self):
        super(ETA_MLP, self).__init__()
        self.route_emb = torch.nn.Embedding(*EMBEDDING_DIMS['route_id'])
        self.node_emb = torch.nn.Embedding(*EMBEDDING_DIMS['node_id'])
        self.weekday_emb = torch.nn.Embedding(*EMBEDDING_DIMS['weekday'])

        self.fc1 = torch.nn.Linear(INPUT_DIM + 8 + 16 + 2, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, route_id, node_id, weekday, dense_feats):
        route_emb = self.route_emb(route_id)
        node_emb = self.node_emb(node_id)
        weekday_emb = self.weekday_emb(weekday)
        x = torch.cat([dense_feats, route_emb, node_emb, weekday_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# postprocess 함수
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

def postprocess_eta_table(eta_table, baseline_table, realtime_raw_dir, yesterday_str):
    for stdid_hhmm, stops in baseline_table.items():
        if stdid_hhmm not in eta_table:
            eta_table[stdid_hhmm] = {}
        for ord_str, time_val in stops.items():
            if ord_str not in eta_table[stdid_hhmm]:
                eta_table[stdid_hhmm][ord_str] = time_val

    stdid_folders = os.listdir(realtime_raw_dir)
    with Pool(cpu_count()) as pool:
        results = pool.map(process_std_folder, [(stdid, REALTIME_BUS_DIR, YESTERDAY_STR) for stdid in stdid_folders])

    for partial_table in results:
        for stdid_hhmm, stops in partial_table.items():
            if stdid_hhmm not in eta_table:
                eta_table[stdid_hhmm] = {}
            eta_table[stdid_hhmm].update(stops)

    return eta_table

def load_forecast(target_date):
    forecast_files = os.listdir(FORECAST_DIR)
    available_files = [f for f in forecast_files if f.endswith('.json')]
    target_forecast = None

    # 가장 가까운 예보 찾기
    for file in sorted(available_files):
        forecast_path = os.path.join(FORECAST_DIR, file)
        forecast_date = datetime.strptime(file.replace('.json', ''), "%Y%m%d")
        if forecast_date <= target_date:
            target_forecast = forecast_path

    if target_forecast:
        with open(target_forecast, 'r') as f:
            return json.load(f)
    return None

def main():
    start_time = time.time()
    print(f"ETA Table 생성 시작... (target={TARGET_STR})")

    df = pd.read_parquet(PARQUET_PATH)

    # 여기 맨뒤에꺼 지우고 넣고로 6/7 조절 가능.
    # 'actual_elapsed_from_departure'
    feature_cols = ['departure_time_sin', 'departure_time_cos', 'departure_time_group', 'PTY', 'RN1', 'T1H', 'actual_elapsed_from_departure']
    X_dense = df[feature_cols].values
    route_id_list = df['route_id'].values
    node_id_list = df['node_id'].values
    weekday_list = df['weekday_encoded'].values
    baseline_elapsed_list = df['baseline_elapsed'].values
    stop_ord_list = df['stop_ord'].values
    departure_hhmm_list = df['departure_hhmm'].values

    X_dense = torch.tensor(X_dense, dtype=torch.float32).to(DEVICE)
    route_id_encoded = torch.tensor(df['route_id_encoded'].values, dtype=torch.long).to(DEVICE)
    node_id_encoded = torch.tensor(df['node_id_encoded'].values, dtype=torch.long).to(DEVICE)
    weekday_encoded = torch.tensor(df['weekday_encoded'].values, dtype=torch.long).to(DEVICE)

    model = ETA_MLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred_delta = model(route_id_encoded, node_id_encoded, weekday_encoded, X_dense).cpu().numpy()

    with open(BASELINE_PATH, 'r') as f:
        baseline_table = json.load(f)

    eta_table = {}
    forecast_data = load_forecast(TARGET_DATE) if USE_FORECAST else None

    for idx in range(len(df)):
        route_id = route_id_list[idx]
        departure_hhmm = departure_hhmm_list[idx]
        stop_ord = str(int(stop_ord_list[idx]))

        stdid_candidates = [stdid for stdid, rname in stdid_number.items() if rname == route_id]
        if not stdid_candidates:
            continue

        stdid = stdid_candidates[0]
        stdid_hhmm = f"{stdid}_{departure_hhmm:04d}"

        dep_hour = departure_hhmm // 100
        dep_min = departure_hhmm % 100
        dep_time = datetime(TARGET_DATE.year, TARGET_DATE.month, TARGET_DATE.day, dep_hour, dep_min, 0)

        dep_seconds = dep_hour * 3600 + dep_min * 60
        baseline_elapsed = baseline_elapsed_list[idx] - dep_seconds
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