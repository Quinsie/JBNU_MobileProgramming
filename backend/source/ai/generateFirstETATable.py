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
YESTERDAY_DATE = datetime(2025, 4, 25) - timedelta(days=1)  # 2025-04-24
YESTERDAY_STR = YESTERDAY_DATE.strftime("%Y%m%d")

PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY_STR}.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model", f"{YESTERDAY_STR}.pth")
BASELINE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{(YESTERDAY_DATE - timedelta(days=1)).strftime('%Y%m%d')}.json")
SAVE_JSON_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY_STR}.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
with open(STDID_NUMBER_PATH, 'r') as f:
    stdid_number = json.load(f)

INPUT_DIM = 9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 정의
class ETA_MLP(torch.nn.Module):
    def __init__(self):
        super(ETA_MLP, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_DIM, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # 데이터 로드
    df = pd.read_parquet(PARQUET_PATH)
    start_time = time.time()

    feature_cols = ['route_id_encoded', 'node_id_encoded', 'stop_ord',
                    'departure_time_sin', 'departure_time_cos',
                    'weekday', 'PTY', 'RN1', 'T1H']

    X = df[feature_cols].values
    baseline_elapsed_list = df['baseline_elapsed'].values
    route_id_list = df['route_id'].values
    stop_ord_list = df['stop_ord'].values

    # departure_hhmm 계산
    departure_hhmm_list = df['departure_hhmm'].values

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    # 모델 로드
    model = ETA_MLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 예측
    with torch.no_grad():
        pred_delta = model(X).squeeze().cpu().numpy()
    
    # debug!
    print(pred_delta[:20])

    # baseline 불러오기
    with open(BASELINE_PATH, 'r') as f:
        baseline_table = json.load(f)

    # ETA Table 생성
    eta_table = {}

    for idx in range(len(df)):
        route_id = route_id_list[idx]
        departure_hhmm = departure_hhmm_list[idx]
        stop_ord = str(int(stop_ord_list[idx]))

        # route_id를 통해 stdid 찾기
        stdid_candidates = [stdid for stdid, rname in stdid_number.items() if rname == route_id]

        if not stdid_candidates:
            continue  # 매칭 실패 시 그냥 스킵 (혹시라도 문제 생길때 대비)

        stdid = stdid_candidates[0]
        stdid_hhmm = f"{stdid}_{departure_hhmm:04d}"

        baseline_elapsed = baseline_elapsed_list[idx]
        delta = pred_delta[idx]
        final_elapsed = baseline_elapsed + delta

        if final_elapsed < 0:
            final_elapsed = 0

        # 출발 시간 기반 ETA 계산
        dep_hour = departure_hhmm // 100
        dep_min = departure_hhmm % 100
        dep_time = datetime(2025, 4, 24, dep_hour, dep_min, 0)
        eta_time = dep_time + timedelta(seconds=final_elapsed)
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