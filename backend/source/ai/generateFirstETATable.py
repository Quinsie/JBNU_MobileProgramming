# backend/source/ai/generateFirstETATable.py

import os
import sys
import json
import torch
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# 설정
# TODAY = datetime.now()
TODAY = datetime(2025, 4, 25)
YESTERDAY_DATE = TODAY - timedelta(days=1)  # 4/24 기준
YESTERDAY_STR = YESTERDAY_DATE.strftime("%Y%m%d")

PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY_STR}.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model", f"{YESTERDAY_STR}.pth")
BASELINE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{(YESTERDAY_DATE - timedelta(days=1)).strftime('%Y%m%d')}.json")
SAVE_JSON_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY_STR}.json")

STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
with open(STDID_NUMBER_PATH, 'r') as f:
    stdid_number = json.load(f)

INPUT_DIM = 6
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

def main():
    start_time = time.time()
    print(f"ETA Table 생성 시작...")

    # 데이터 로드
    df = pd.read_parquet(PARQUET_PATH)

    feature_cols = ['departure_time_sin', 'departure_time_cos', 'departure_time_group', 'PTY', 'RN1', 'T1H']
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

    # baseline 로드
    with open(BASELINE_PATH, 'r') as f:
        baseline_table = json.load(f)

    # ETA Table 생성
    eta_table = {}

    for idx in range(len(df)):
        route_id = route_id_list[idx]
        departure_hhmm = departure_hhmm_list[idx]
        stop_ord = str(int(stop_ord_list[idx]))

        # stdid 찾기
        stdid_candidates = [stdid for stdid, rname in stdid_number.items() if rname == route_id]
        if not stdid_candidates:
            continue

        stdid = stdid_candidates[0]
        stdid_hhmm = f"{stdid}_{departure_hhmm:04d}"

        baseline_elapsed = baseline_elapsed_list[idx]
        delta = pred_delta[idx]
        final_elapsed = baseline_elapsed + delta

        if final_elapsed < 0:
            final_elapsed = 0

        # 정확한 ETA 계산
        dep_hour = departure_hhmm // 100
        dep_min = departure_hhmm % 100
        dep_time = datetime(YESTERDAY_DATE.year, YESTERDAY_DATE.month, YESTERDAY_DATE.day, dep_hour, dep_min, 0)
        # ETA 계산
        eta_time = dep_time + timedelta(seconds=int(final_elapsed))
        eta_time_str = eta_time.strftime("%H:%M:%S")

        if stdid_hhmm not in eta_table:
            eta_table[stdid_hhmm] = {}
        eta_table[stdid_hhmm][stop_ord] = eta_time_str

    # 저장
    os.makedirs(os.path.dirname(SAVE_JSON_PATH), exist_ok=True)
    with open(SAVE_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(eta_table, f, indent=2, ensure_ascii=False)

    print(f"ETA Table 생성 완료: {SAVE_JSON_PATH}")
    print("총 소요시간: ", time.time() - start_time, "sec")

if __name__ == "__main__":
    main()