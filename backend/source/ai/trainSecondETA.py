# backend/source/ai/trainSecondETA.py

import os
import sys
import time
import torch
import argparse
import pandas as pd
import torch.nn as nn
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
from SecondETAModel import SecondETAModel

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "data", "model", "secondETA")
MODEL_SAVE_PATH_1 = None
MODEL_SAVE_PATH_2 = None
SELF_REVIEW_PATH = None
REPLAY_PATH = None
YESTERDAY_MODEL_PATH_2 = None

# === 하이퍼파라미터 ===
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001

# === 디바이스 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ZipDataset(Dataset):
    def __init__(self, x_dict, y, mask):
        self.x_dict = x_dict
        self.y = y
        self.mask = mask
        self.keys = list(x_dict.keys())
        self.length = y.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_vals = [self.x_dict[key][idx] for key in self.keys]
        return idx, *x_vals, self.y[idx], self.mask[idx]

def train_model(phase: str):
    # === phase에 따라 학습용 parquet 경로 선택 ===
    if phase == "self_review":
        parquet_path = SELF_REVIEW_PATH
        model_load_path = YESTERDAY_MODEL_PATH_2
        model_save_path = MODEL_SAVE_PATH_1
    elif phase == "replay":
        parquet_path = REPLAY_PATH
        model_load_path = MODEL_SAVE_PATH_1
        model_save_path = MODEL_SAVE_PATH_2
    else:
        raise ValueError("Invalid phase. Use 'self_review' or 'replay'.")

    print(f"[INFO] Loading {phase} dataset: {parquet_path}")
    df = pd.read_parquet(parquet_path)  # parquet 파일 읽기

    # === 모델 정의 및 전날 모델 불러오기 ===
    model = SecondETAModel().to(device)
    if os.path.exists(model_load_path):
        print(f"[INFO] Loading previous model from: {model_load_path}")
        model.load_state_dict(torch.load(model_load_path, map_location=device))
    else:
        print(f"[INFO] No previous model found at {model_load_path}. Initializing new model.")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # x_로 시작하는 모든 column을 feature로 간주
    x_cols = [col for col in df.columns if col.startswith("x_")]
    mask_cols = [f"mask_{i}" for i in range(1, 6)]  # 마스킹 열 추가

    # self_review일 때는 prev_pred_elapsed가 반드시 있어야 함
    if phase == "self_review":
        for i in range(1, 6):
            assert f"x_prev_pred_elapsed_{i}" in x_cols

    # 딕셔너리로 feature 구성
    x_dict = {}
    for col in x_cols:
        key = col.replace("x_", "")
        
        # torch dtype 지정
        if df[col].dtype in ["int64", "int32"]:
            tensor = torch.tensor(df[col].values, dtype=torch.long)
        else:
            tensor = torch.tensor(df[col].values, dtype=torch.float32)
        
        # [B] → [B, 1] 변환
        # if tensor.dim() == 1 and tensor.dtype == torch.float32:
        #     tensor = tensor.unsqueeze(1)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(1)

        x_dict[key] = tensor.to(device)

    y_cols = [f"y_{i}" for i in range(1, 6)]
    y = torch.tensor(df[y_cols].values, dtype=torch.float32).to(device)
    mask = torch.tensor(df[mask_cols].values, dtype=torch.float32).to(device)  # shape: (B, 10)

    # dataset은 리스트(zip) 기반으로 구성
    dataset = ZipDataset(x_dict, y, mask)
    keys = list(x_dict.keys())

    # === 학습 루프 ===
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
            _, *x_vals, batch_y, batch_mask = batch
            batch_x = dict(zip(keys, x_vals))

            # debug
            # for key in batch_x:
            #     try:
            #         print(f"[DEBUG] {key} min: {batch_x[key].min().item()}, max: {batch_x[key].max().item()}")
            #     except Exception as e:
            #         print(f"[DEBUG] {key} → skip ({e})")

            optimizer.zero_grad()

            # forward pass
            pred_mean, pred_log_var = model(batch_x)

            # 기본 heteroscedastic loss 계산
            # ORD+1 ~ ORD+5 가중치: 앞쪽이 더 중요
            weights = [3.0, 2.0, 1.5, 1.2, 1.0]  # 완만한 감소

            hetero_loss = 0
            for i in range(5):
                diff_sq = (batch_y[:, i] - pred_mean[:, i]) ** 2
                log_var = pred_log_var[:, i]
                loss_i = diff_sq * torch.exp(-log_var) + log_var
                loss_i = loss_i * batch_mask[:, i]
                hetero_loss += weights[i] * (loss_i.sum() / (batch_mask[:, i].sum() + 1e-6))

            loss = hetero_loss / sum(weights)  # 평균화

            # === self-review residual penalty ===
            if phase == "self_review":
                prev_pred = torch.cat([batch_x[f"prev_pred_elapsed_{i}"].detach() for i in range(1, 6)], dim=1)
                penalty = nn.functional.relu(batch_y - prev_pred) * batch_mask
                penalty = penalty.sum() / (batch_mask.sum() + 1e-6)
                loss = hetero_loss + 0.3 * penalty
                # print(f"[E{epoch+1}] hetero_loss: {hetero_loss.item():.4f} | penalty: {penalty.item():.4f}") # DEBUG

            # pred_mean이 (B, 5) → ORD+1 ~ ORD+5에 대해 순서 보장
            ranking_diff = nn.functional.relu(pred_mean[:, :-1] - pred_mean[:, 1:])         # (B, 4)
            mask_pairwise = batch_mask[:, :-1] * batch_mask[:, 1:]                          # (B, 4)
            ranking_loss = (ranking_diff * mask_pairwise).sum() / (mask_pairwise.sum() + 1e-6)
            loss += 0.2 * ranking_loss

            # print(f"[E{epoch+1}] pred_mean: min={pred_mean.min().item():.4f}, max={pred_mean.max().item():.4f}, mean={pred_mean.mean().item():.4f}") # DEBUG

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # 에폭당 평균 loss와 최대 예측값 출력
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataset):.4f} | pred_mean: min={pred_mean.min().item():.4f}, max={pred_mean.max().item():.4f}, mean={pred_mean.mean().item():.4f}")

    # === 모델 저장 ===
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Saved model to {model_save_path}")

    # === test ===
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=True)))
        _, *x_vals, batch_y, batch_mask = test_batch
        batch_x = dict(zip(keys, x_vals))

        pred_mean, _ = model(batch_x)
        pred_mean = pred_mean.masked_fill(batch_mask == 0, 0.0)
        pred_elapsed = pred_mean.squeeze() * 3000
        real_elapsed = batch_y.squeeze() * 3000

        print("\n=== 학습 후 추론 테스트 (pred vs real) ===")
        for i in range(min(10, len(pred_elapsed))):
            pred_vals = ", ".join([f"{v:.2f}" for v in pred_elapsed[i].tolist()])
            real_vals = ", ".join([f"{v:.2f}" for v in real_elapsed[i].tolist()])
            print(f"[{i}] Pred: [{pred_vals}] | Real: [{real_vals}]")

# === main 함수 진입점 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["self_review", "replay"], help="학습 단계: self_review 또는 replay")
    parser.add_argument("--date", required=True, help="학습 날짜 (YYYYMMDD)")
    args = parser.parse_args()

    DATE = args.date
    YESTERDAY = (datetime.strptime(DATE, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    
    MODEL_SAVE_PATH_1 = os.path.join(MODEL_DIR, "self_review", f"{DATE}.pth")
    MODEL_SAVE_PATH_2 = os.path.join(MODEL_DIR, "replay", f"{DATE}.pth")
    SELF_REVIEW_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "second_train", "self_review", f"{DATE}.parquet")
    REPLAY_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "second_train", "replay", f"{DATE}.parquet")
    YESTERDAY_MODEL_PATH_2 = os.path.join(MODEL_DIR, "replay", f"{YESTERDAY}.pth")

    now = time.time()
    train_model(args.mode)
    print("총 학습 시간: ", time.time() - now, "sec")