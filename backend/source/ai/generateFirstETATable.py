# backend/source/ai/generateFirstETATable.py

import os
import sys
import json
import torch
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# 설정
# TODAY = datetime.now()
TODAY = datetime(2025, 4, 25)
YESTERDAY_DATE = TODAY - timedelta(days=1)
YESTERDAY_STR = YESTERDAY_DATE.strftime("%Y%m%d")

PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY_STR}.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model", f"{YESTERDAY_STR}.pth")
BASELINE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{(YESTERDAY_DATE - timedelta(days=1)).strftime('%Y%m%d')}.json")
SAVE_JSON_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY_STR}.json")
REALTIME_BUS_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "realtime_bus")

STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
with open(STDID_NUMBER_PATH, 'r') as f:
    stdid_number = json.load(f)

INPUT_DIM = 7
EMBEDDING_DIMS = {
    'route_id': (500, 8),
    'node_id': (3200, 16),
    'weekday': (3, 2),
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델
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

def postprocess_eta_table(eta_table, baseline_table, realtime_raw_dir, yesterday_str):
    # 1. baseline 기반 누락 복구
    for stdid_hhmm, stops in baseline_table.items():
        if stdid_hhmm not in eta_table:
            eta_table[stdid_hhmm] = {}
        for ord_str, time_val in stops.items():
            if ord_str not in eta_table[stdid_hhmm]:
                eta_table[stdid_hhmm][ord_str] = time_val

    # 2. raw 기반 누락 복구
    def process_std_folder(stdid_folder):
        recovered = {}
        stdid_path = os.path.join(realtime_raw_dir, stdid_folder)
        if not os.path.isdir(stdid_path):
            return recovered

        for file in os.listdir(stdid_path):
            if not file.startswith(yesterday_str):
                continue

            hhmm = file.split('_')[-1].split('.')[0]
            stdid_hhmm = f"{stdid_folder}_{hhmm}"

            if stdid_hhmm not in eta_table:
                eta_table[stdid_hhmm] = {}

            file_path = os.path.join(stdid_path, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                logs = data.get("stop_reached_logs", [])
                for log in logs:
                    ord_str = str(log['ord'])
                    time_val = log['time'][-8:]
                    if ord_str not in eta_table[stdid_hhmm]:
                        recovered.setdefault(stdid_hhmm, {})[ord_str] = time_val
            except Exception as e:
                continue
        return recovered

    stdid_folders = os.listdir(realtime_raw_dir)
    with Pool(cpu_count()) as pool:
        results = pool.map(process_std_folder, stdid_folders)

    for partial_table in results:
        for stdid_hhmm, stops in partial_table.items():
            if stdid_hhmm not in eta_table:
                eta_table[stdid_hhmm] = {}
            eta_table[stdid_hhmm].update(stops)

    return eta_table

def main():
    start_time = time.time()
    print(f"ETA Table 생성 시작...")

    # 데이터 로드
    df = pd.read_parquet(PARQUET_PATH)

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

    # 모델 로드
    model = ETA_MLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        pred_delta = model(route_id_encoded, node_id_encoded, weekday_encoded, X_dense).cpu().numpy()

    with open(BASELINE_PATH, 'r') as f:
        baseline_table = json.load(f)

    eta_table = {}

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
        dep_time = datetime(YESTERDAY_DATE.year, YESTERDAY_DATE.month, YESTERDAY_DATE.day, dep_hour, dep_min, 0)

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

    # 보정
    eta_table = postprocess_eta_table(eta_table, baseline_table, REALTIME_BUS_DIR, YESTERDAY_STR)

    # 저장
    os.makedirs(os.path.dirname(SAVE_JSON_PATH), exist_ok=True)
    with open(SAVE_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(eta_table, f, indent=2, ensure_ascii=False)

    print(f"ETA Table 생성 완료: {SAVE_JSON_PATH}")
    print("총 소요시간: ", time.time() - start_time, "sec")

if __name__ == "__main__":
    main()