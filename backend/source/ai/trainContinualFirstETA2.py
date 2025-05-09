# backend/source/ai/trainContinualFirstETA.py

import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# 설정
# TODAY = datetime.now()
TODAY = datetime(2025, 4, 25)
YESTERDAY_DATE = TODAY - timedelta(days=1)  # 4/24 기준
YESTERDAY_STR = YESTERDAY_DATE.strftime("%Y%m%d")

PARQUET_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", f"{YESTERDAY_STR}.parquet")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "data", "model", f"{YESTERDAY_STR}.pth")

INPUT_DIM = 6  # Dense로 들어갈 feature 개수
EMBEDDING_DIMS = {
    'route_id': (500, 8),
    'node_id': (3200, 16),
    'weekday_timegroup': (24, 4),  # ✅ 수정됨
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 600
BATCH_SIZE = 2048
LEARNING_RATE = 0.0005

# Dataset
class ETADataset(Dataset):
    def __init__(self, df):
        self.route_id = df['route_id_encoded'].values
        self.node_id = df['node_id_encoded'].values
        self.weekday_timegroup = df['weekday_timegroup_encoded'].values  # ✅ 수정
        self.dense_feats = df[['departure_time_sin', 'departure_time_cos', 'departure_time_group',
                               'PTY', 'RN1', 'T1H']].values
        self.targets = df['delta_elapsed'].values

    def __getitem__(self, idx):
        return (
            torch.tensor(self.route_id[idx], dtype=torch.long),
            torch.tensor(self.node_id[idx], dtype=torch.long),
            torch.tensor(self.weekday_timegroup[idx], dtype=torch.long),  # ✅ 수정
            torch.tensor(self.dense_feats[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# Model
class ETA_MLP(nn.Module):
    def __init__(self):
        super(ETA_MLP, self).__init__()
        self.route_emb = nn.Embedding(*EMBEDDING_DIMS['route_id'])
        self.node_emb = nn.Embedding(*EMBEDDING_DIMS['node_id'])
        self.weekday_timegroup_emb = nn.Embedding(*EMBEDDING_DIMS['weekday_timegroup'])  # ✅ 수정

        self.fc1 = nn.Linear(INPUT_DIM + 8 + 16 + 4, 128)  # ✅ 수정
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, route_id, node_id, weekday_timegroup, dense_feats):  # ✅ 수정
        route_emb = self.route_emb(route_id)
        node_emb = self.node_emb(node_id)
        weekday_emb = self.weekday_timegroup_emb(weekday_timegroup)  # ✅ 수정

        x = torch.cat([dense_feats, route_emb, node_emb, weekday_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# 학습 함수
def train():
    df = pd.read_parquet(PARQUET_PATH)
    start_time = time.time()

    dataset = ETADataset(df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = ETA_MLP().to(DEVICE)

    # 전날 모델 경로
    DAY_BEFORE_MODEL_PATH = os.path.join(BASE_DIR, "data", "model", f"{(YESTERDAY_DATE - timedelta(days=1)).strftime('%Y%m%d')}.pth")

    # 전날 모델이 존재하면 불러오기
    if os.path.exists(DAY_BEFORE_MODEL_PATH):
        print(f"전날 모델 불러오기: {DAY_BEFORE_MODEL_PATH}")
        model.load_state_dict(torch.load(DAY_BEFORE_MODEL_PATH, map_location=DEVICE))
    else:
        print(f"전날 모델 없음. 새로운 모델로 학습 시작.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for route_id, node_id, weekday_timegroup, dense_feats, targets in dataloader:  # ✅ 수정
            route_id = route_id.to(DEVICE)
            node_id = node_id.to(DEVICE)
            weekday_timegroup = weekday_timegroup.to(DEVICE)  # ✅ 수정
            dense_feats = dense_feats.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(route_id, node_id, weekday_timegroup, dense_feats)  # ✅ 수정
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * route_id.size(0)

        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataset):.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"모델 저장 완료: {MODEL_SAVE_PATH}")
    print("총 소요시간: ", time.time() - start_time, "sec")

if __name__ == "__main__":
    train()