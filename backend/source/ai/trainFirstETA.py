# backend/source/ai/trainFirstETA.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import joblib

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# 설정
PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", "20250424.parquet")
ROUTE_ENCODER_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "route_encoder.pkl")
NODE_ENCODER_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "node_encoder.pkl")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "data", "model", "20250424.pth")

INPUT_DIM = 9
BATCH_SIZE = 1024
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MLP 모델 정의
class ETA_MLP(nn.Module):
    def __init__(self):
        super(ETA_MLP, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIM, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    # 데이터 불러오기
    df = pd.read_parquet(PARQUET_PATH)

    feature_cols = ['route_id_encoded', 'node_id_encoded', 'stop_ord',
                    'departure_time_sin', 'departure_time_cos',
                    'weekday', 'PTY', 'RN1', 'T1H']
    target_col = 'delta_elapsed'

    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 모델, 손실함수, 옵티마이저 세팅
    model = ETA_MLP().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 학습 시작
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(loader.dataset)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

    # 모델 저장
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ 모델 저장 완료: {MODEL_SAVE_PATH}")

    # (선택) LabelEncoder 저장 - 이건 나중 추론 때 쓰려고
    if os.path.exists(ROUTE_ENCODER_PATH) and os.path.exists(NODE_ENCODER_PATH):
        print("✅ 이미 인코더 저장되어 있음.")
    else:
        print("⚠️ 인코더 저장은 별도로 진행 필요.")

if __name__ == "__main__":
    main()