# backend/source/ai/trainFirstETAContinual.py

import os
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.logger import log

def train_first_eta_continual():
    # 날짜 설정
    today = datetime.now().date()
    # today = datetime(2025, 4, 25).date()  # << 수동 지정시 여기 활성화
    yesterday = today - timedelta(days=1)
    two_days_ago = today - timedelta(days=2)
    YESTERDAY = yesterday.strftime("%Y%m%d")
    TWO_DAYS_AGO = two_days_ago.strftime("%Y%m%d")

    PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", f"{YESTERDAY}.parquet")
    PREV_MODEL_PATH = os.path.join(BASE_DIR, "data", "models", "firstETA", f"{TWO_DAYS_AGO}.pth")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "data", "models", "firstETA", f"{YESTERDAY}.pth")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("trainFirstETA", f"Using device: {device}")

    # 데이터 불러오기
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        log("trainFirstETA", f"Parquet 불러오기 실패: {e}")
        return

    # 결측치 처리
    df["PTY"] = df["PTY"].fillna(0)
    df["RN1"] = df["RN1"].fillna(0)
    df["T1H"] = df["T1H"].fillna(0)

    # 라벨 인코딩 (route_id)
    le = LabelEncoder()
    df["route_id_encoded"] = le.fit_transform(df["route_id"])

    # Feature, Target 분리
    feature_cols = ["route_id_encoded", "departure_time", "day_type", "stop_order", "PTY", "RN1", "T1H"]
    X = df[feature_cols].values
    y = df["target_elapsed_time"].values

    # train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Tensor 변환
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # MLP 모델 정의
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

    model = ETA_MLP(input_dim=X_train.shape[1]).to(device)

    # 이전 모델 불러오기
    if os.path.exists(PREV_MODEL_PATH):
        checkpoint = torch.load(PREV_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        prev_label_encoder = checkpoint["label_encoder"]
        log("trainFirstETA", f"이전 모델 로드 완료: {PREV_MODEL_PATH}")
    else:
        log("trainFirstETA", f"이전 모델 없음 → 새 모델로 시작")

    # Loss, Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 학습 시작
    start_time = time.time()
    log("trainFirstETA", f"{YESTERDAY} ETA 추가 학습 시작")

    epochs = 300
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        if epoch % 10 == 0:
            log("trainFirstETA", f"[Epoch {epoch}] Train Loss: {loss.item():.4f} / Val Loss: {val_loss.item():.4f}")

    # 모델 저장
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_encoder": le,
    }, MODEL_SAVE_PATH)

    elapsed = time.time() - start_time
    log("trainFirstETA", f"모델 저장 완료: {MODEL_SAVE_PATH}")
    log("trainFirstETA", f"총 추가 학습 소요시간: {elapsed:.2f}초")

if __name__ == "__main__":
    train_first_eta_continual()