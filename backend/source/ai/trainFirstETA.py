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
    dataset = list(zip(*(list(x_dict.values()) + [y])))
    keys = list(x_dict.keys())

    # === 학습 루프 ===
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True):
            *x_vals, batch_y = batch
            batch_x = dict(zip(keys, x_vals))
            
            optimizer.zero_grad()

            # forward pass
            pred_mean, pred_log_var = model(batch_x, phase=phase)

            # 기본 heteroscedastic loss 계산
            hetero_loss = ((batch_y - pred_mean) ** 2 * torch.exp(-pred_log_var) + pred_log_var).mean()
            loss = hetero_loss

            # self-review의 경우 residual penalty 추가
            if phase == "self_review":
                prev_pred = batch_x["prev_pred_elapsed"].detach().unsqueeze(1)
                penalty = nn.functional.relu(batch_y - prev_pred).mean()
                loss = hetero_loss + 0.3 * penalty
            
            # ranking loss 추가 (순서 보장)
            if "trip_group_id" in df.columns and "ord" in df.columns:
                trip_to_indices = defaultdict(list)
                for idx, trip in enumerate(df["trip_group_id"].values):
                    trip_to_indices[trip].append(idx)
                
                ranking_loss = 0
                count = 0
                for indices in trip_to_indices.values():
                    if len(indices) < 2:
                        continue

                    # 1. ord 기준으로 정렬된 index 리스트 얻기
                    sorted_indices = sorted(indices, key=lambda idx: df["ord"].iloc[idx])

                    # 2. 정렬된 ord와 예측값 가져오기
                    ords = torch.tensor(df["ord"].values[sorted_indices], dtype=torch.float32).to(device)
                    preds = pred_mean[sorted_indices].squeeze()

                    # 3. 순서대로 loss 계산 (단, 같은 ord는 무시)
                    for i in range(len(ords) - 1):
                        if ords[i] < ords[i + 1]:  # 동률은 스킵
                            ranking_loss += nn.functional.relu(preds[i] - preds[i + 1])
                            count += 1     
                            
                if count > 0:
                    ranking_loss = ranking_loss / count
                    loss += 0.1 * ranking_loss
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 에폭당 평균 loss와 최대 예측값 출력
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataset):.4f}, Max pred: {pred_mean.max().item():.2f}")

    # === 모델 저장 ===
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"[INFO] Saved model to {model_save_path}")

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