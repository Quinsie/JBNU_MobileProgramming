# backend/source/ai/generateFirstETATable.py

import os
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.logger import log

class ETA_MLP(nn.Module):
    def __init__(self, input_dim):
        super(ETA_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_eta_table():
    # 날짜 설정
    today = datetime.now().date()
    # today = datetime(2025, 4, 25).date()
    yesterday = today - timedelta(days=1)

    YESTERDAY = yesterday.strftime("%Y%m%d")

    PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", f"{YESTERDAY}.parquet")
    MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "firstETA", f"{YESTERDAY}.pth")
    STDID_MAP_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")
    SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", f"{YESTERDAY}.json")
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("generateETATable", f"Using device: {device}")

    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        log("generateETATable", f"Parquet 불러오기 실패: {e}")
        return

    if not os.path.exists(MODEL_PATH):
        log("generateETATable", f"모델 파일 없음: {MODEL_PATH}")
        return

    try:
        with open(STDID_MAP_PATH, encoding="utf-8") as f:
            stdid_to_route = json.load(f)
    except Exception as e:
        log("generateETATable", f"STDID 매핑파일 불러오기 실패: {e}")
        return

    route_to_stdid = {}
    for stdid, route_id in stdid_to_route.items():
        route_to_stdid.setdefault(route_id, []).append(stdid)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = ETA_MLP(input_dim=7).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    le = checkpoint["label_encoder"]

    # 전처리
    df["PTY"] = df["PTY"].fillna(0)
    df["RN1"] = df["RN1"].fillna(0)
    df["T1H"] = df["T1H"].fillna(0)
    df["route_id_encoded"] = le.transform(df["route_id"])

    # ETA Table 저장용 dict
    eta_table = {}

    # 🎯 최적화 추론 시작
    start_time = time.time()
    log("generateETATable", f"{YESTERDAY} ETA Table 생성 시작")

    feature_cols = ["route_id_encoded", "departure_time", "day_type", "stop_order", "PTY", "RN1", "T1H"]
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_elapsed_batch = model(features).squeeze(1).cpu().numpy()

    for idx, row in df.iterrows():
        route_id = row["route_id"]
        departure_time = int(row["departure_time"])
        stop_order = str(row["stop_order"])
        pred_elapsed = pred_elapsed_batch[idx]

        stdid_list = route_to_stdid.get(route_id)
        if not stdid_list:
            log("generateETATable", f"STDID 매칭 실패: {route_id}")
            sys.exit(1)

        stdid = stdid_list[0]

        # 시간 계산
        dep_hour = departure_time // 60
        dep_minute = departure_time % 60
        base_seconds = dep_hour * 3600 + dep_minute * 60

        arrival_seconds = max(base_seconds + pred_elapsed, base_seconds)

        arr_hour = int(arrival_seconds // 3600) % 24
        arr_minute = int((arrival_seconds % 3600) // 60)
        arr_second = int(arrival_seconds % 60)

        arr_time_str = f"{arr_hour:02d}:{arr_minute:02d}:{arr_second:02d}"

        key = f"{stdid}_{departure_time:04d}"
        if key not in eta_table:
            eta_table[key] = {}
        eta_table[key][stop_order] = arr_time_str

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(eta_table, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    log("generateETATable", f"ETA Table 저장 완료 → {SAVE_PATH}")
    log("generateETATable", f"총 소요시간: {elapsed:.2f}초")

if __name__ == "__main__":
    generate_eta_table()