# backend/source/ai/trainFirstETA.py

import os
import sys
import time
import torch
import argparse
import pandas as pd
import torch.nn as nn
from collections import defaultdict
from datetime import datetime, timedelta
from torch.utils.data import DataLoader

# === 경로 설정 ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")); sys.path.append(BASE_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "data", "model", "firstETA")
ODEL_SAVE_PATH_1 = None
MODEL_SAVE_PATH_2 = None
SELF_REVIEW_PATH = None
REPLAY_PATH = None
YESTERDAY_MODEL_PATH_2 = None
from source.ai.FirstETAModel import FirstETAModel

# === 하이퍼파라미터 ===
EPOCHS = 15
BATCH_SIZE = 512
LR = 0.001

# === 디바이스 설정 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model = FirstETAModel().to(device)
    if os.path.exists(model_load_path):
        print(f"[INFO] Loading previous model from: {model_load_path}")
        model.load_state_dict(torch.load(model_load_path, map_location=device))
    else:
        print(f"[INFO] No previous model found at {model_load_path}. Initializing new model.")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # x_로 시작하는 모든 column을 feature로 간주
    x_cols = [col for col in df.columns if col.startswith("x_")]

    # self_review일 때는 prev_pred_elapsed가 반드시 있어야 함
    if phase == "self_review":
        assert "x_prev_pred_elapsed" in x_cols, "self_review인데 x_prev_pred_elapsed가 없음"

    y_col = "y"

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
        if tensor.dim() == 1 and tensor.dtype == torch.float32:
            tensor = tensor.unsqueeze(1)

        x_dict[key] = tensor.to(device)

    y = torch.tensor(df[y_col].values, dtype=torch.float32).unsqueeze(1).to(device)

    # dataset은 리스트(zip) 기반으로 구성
    dataset = list(zip(range(len(df)), *(list(x_dict.values()) + [y])))
    keys = list(x_dict.keys())

    # === 학습 루프 ===
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
            batch_indices, *x_vals, batch_y = batch
            batch_x = dict(zip(keys, x_vals))

            optimizer.zero_grad()

            # forward pass
            pred_mean, pred_log_var = model(batch_x)

            # 기본 heteroscedastic loss 계산
            hetero_loss = ((batch_y - pred_mean) ** 2 * torch.exp(-pred_log_var) + pred_log_var).mean()
            loss = hetero_loss

            # === self-review residual penalty ===
            if phase == "self_review":
                prev_pred = batch_x["prev_pred_elapsed"].detach().unsqueeze(1)
                penalty = nn.functional.relu(batch_y - prev_pred).mean()
                loss = hetero_loss + 0.3 * penalty
                # print(f"[E{epoch+1}] hetero_loss: {hetero_loss.item():.4f} | penalty: {penalty.item():.4f}") # DEBUG


            # === [여기에 ranking loss를 이동] ===
            if "trip_group_id" in df.columns and "ord" in df.columns:
                # 각 배치 인덱스에 대한 trip_group_id를 수집
                trip_ids = df["trip_group_id"].values[[i.item() for i in batch_indices]]
                ords = torch.tensor(df["ord"].values[[i.item() for i in batch_indices]], dtype=torch.float32).to(device)
                preds = pred_mean.squeeze()

                trip_to_local_indices = defaultdict(list)
                for local_idx, trip in enumerate(trip_ids):
                    trip_to_local_indices[trip].append(local_idx)

                ranking_loss = 0
                count = 0
                for indices in trip_to_local_indices.values():
                    if len(indices) < 2:
                        continue
                    ord_subset = ords[indices]
                    pred_subset = preds[indices]

                    sorted_idx = torch.argsort(ord_subset)
                    ord_sorted = ord_subset[sorted_idx]
                    pred_sorted = pred_subset[sorted_idx]

                    for i in range(len(ord_sorted) - 1):
                        if ord_sorted[i] < ord_sorted[i + 1]:  # 동률은 스킵
                            ranking_loss += nn.functional.relu(pred_sorted[i] - pred_sorted[i + 1])
                            count += 1

                if count > 0:
                    ranking_loss = ranking_loss / count
                    loss += 0.2 * ranking_loss
                    # print(f"[E{epoch+1}] ranking_loss: {ranking_loss.item():.4f}") # DEBUG

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
        batch_indices, *x_vals, batch_y = test_batch
        batch_x = dict(zip(keys, x_vals))

        pred_mean, _ = model(batch_x)
        pred_elapsed = pred_mean.squeeze() * 7200
        real_elapsed = batch_y.squeeze() * 7200

        print("\n=== 학습 후 추론 테스트 (pred vs real) ===")
        for i in range(min(10, len(pred_elapsed))):
            print(f"[{i}] Pred: {pred_elapsed[i]:.2f}s | Real: {real_elapsed[i]:.2f}s")

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
    SELF_REVIEW_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "self_review", f"{DATE}.parquet")
    REPLAY_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "first_train", "replay", f"{DATE}.parquet")
    YESTERDAY_MODEL_PATH_2 = os.path.join(MODEL_DIR, "replay", f"{YESTERDAY}.pth")

    now = time.time()
    train_model(args.mode)
    print("총 학습 시간: ", time.time() - now, "sec")