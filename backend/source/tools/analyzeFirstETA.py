# backend/source/tools/analyzeFirstETA.py

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 모델 정의 ----------------
class ETA_MLP(nn.Module):
    def __init__(self):
        super(ETA_MLP, self).__init__()
        self.route_emb = nn.Embedding(500, 8)
        self.node_emb = nn.Embedding(3200, 16)
        self.weekday_emb = nn.Embedding(3, 2)
        self.fc1 = nn.Linear(6 + 8 + 16 + 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, route_id, node_id, weekday, dense_feats):
        route_emb = self.route_emb(route_id)
        node_emb = self.node_emb(node_id)
        weekday_emb = self.weekday_emb(weekday)
        x = torch.cat([dense_feats, route_emb, node_emb, weekday_emb], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()

# ---------------- 분석 함수 ----------------
def analyze(model_path, parquet_path):
    os.makedirs("backend/data/processed/analyzed", exist_ok=True)
    
    # 모델 로드
    model = ETA_MLP()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 데이터 로드
    df = pd.read_parquet(parquet_path)
    route = torch.tensor(df['route_id_encoded'].values, dtype=torch.long)
    node = torch.tensor(df['node_id_encoded'].values, dtype=torch.long)
    weekday = torch.tensor(df['weekday_encoded'].values, dtype=torch.long)
    dense = torch.tensor(df[['departure_time_sin', 'departure_time_cos', 'departure_time_group',
                             'PTY', 'RN1', 'T1H']].values, dtype=torch.float32)

    # 예측값 생성
    preds = []
    with torch.no_grad():
        BATCH = 2048
        for i in range(0, len(df), BATCH):
            r = route[i:i+BATCH]
            n = node[i:i+BATCH]
            w = weekday[i:i+BATCH]
            d = dense[i:i+BATCH]
            pred = model(r, n, w, d)
            preds.extend(pred.numpy())
    df["y_pred"] = preds
    df["error"] = np.abs(df["delta_elapsed"] - df["y_pred"])

    # ① 노선별 MSE 시각화
    route_mse = df.groupby("route_id") \
                  .apply(lambda g: mean_squared_error(g["delta_elapsed"], g["y_pred"])) \
                  .sort_values(ascending=False)
    plt.figure(figsize=(20,5))
    route_mse.plot(kind='bar')
    plt.title("노선별 MSE Loss")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig("backend/data/processed/analyzed/route_loss.png")
    print("노선별 MSE 시각화 완료 → route_loss.png")

    # ② 시간대/요일별 평균 오차 Heatmap
    heat = df.pivot_table(index="departure_time_group", columns="weekday", values="error", aggfunc="mean")
    plt.figure(figsize=(6,5))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("시간대 vs 요일 평균 오차")
    plt.tight_layout()
    plt.savefig("backend/data/processed/analyzed/time_weekday_heatmap.png")
    print("시간대/요일별 Heatmap 저장 → time_weekday_heatmap.png")

    # ③ route 임베딩 시각화
    route_emb = model.route_emb.weight.detach().cpu().numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(route_emb)
    plt.figure(figsize=(6,6))
    plt.scatter(reduced[:,0], reduced[:,1], alpha=0.7)
    plt.title("Route Embedding (PCA)")
    plt.tight_layout()
    plt.savefig("backend/data/processed/analyzed/route_embedding_pca.png")
    print("Route Embedding PCA 저장 → route_embedding_pca.png")

if __name__ == "__main__":
    model_path = "backend/data/model/20250425.pth"
    parquet_path = "backend/data/preprocessed/first_train/20250425.parquet"
    analyze(model_path, parquet_path)