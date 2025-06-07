# backend/source/ai/relaySecondModel.py

import os
import sys
import time
import torch
from queue import Empty
from datetime import date, datetime
from multiprocessing import Queue
from multiprocessing.managers import BaseManager

# 실행날짜 기억하고, json파일로 저장은 "당일 날짜"로 진행해야된다.
# 절대 잊지 말 것!!!
# 중계기는 실행과 동시에 해당 날짜의 first eta 예측파일을 second eta 파일로 복사하고, 그걸 계속 수정해야한다.
# 또한 추론을 위한 json 저장은 별도로 진행한다. 얘도 한 파일에 넣고.

# === BASE ENVs SETTING ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

MEAN_NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "node")
WEATHER_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "weather")
TRAFFIC_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "traffic")
ROUTE_NODES_PAIR_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_pair")
ROUTE_NODES_MAPPED_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_mapped")

from source.utils.logger import log
from inferenceSecondModel import infer_eta_batch
from inferenceSecondModel import prepare_input_tensor
from inferenceSecondModel import load_second_eta_model

# === Queue Server Manager Settings ===
q = Queue()

class QueueManager(BaseManager): pass
QueueManager.register("get_queue", callable=lambda: q)

def start_queue_server():
    manager = QueueManager(address=("localhost", 50000), authkey=b"abc")
    server = manager.get_server()
    print("[RELAY] Queue 서버 시작됨 (포트 50000)")
    server.serve_forever()

# === Sub Function ===
def send_to_queue(payload: dict, log_prefix: str):
    try:
        class QueueManager(BaseManager): pass
        QueueManager.register("get_queue")
        manager = QueueManager(address=("localhost", 49000), authkey=b"def")
        manager.connect()
        q = manager.get_queue()
        q.put(payload)
        log("relaySecondModel", f"{log_prefix} 전송 완료: {payload}")
    except Exception as e:
        log("relaySecondModel", f"{log_prefix} 전송 실패: {payload.get('stdid', '?')}_{e}")

# === Process Main Function ===
def consume_loop():
    print("[RELAY] Queue 소비 시작")
    today = date.today() # SET DATE TODAY
    today_str = datetime.strftime(today, "%Y%m%d")

    # = SET INFERENCE MODEL =
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(BASE_DIR, "data", "model", "secondETA", "replay", f"{today_str}.pth")
    model = load_second_eta_model(model_path, device)

    while True:
        try:
            row = q.get(timeout = 1)
            type = row.get("type", 4)
            if type == 0: # route node
                pass 
            elif type == 1: # ord
                pass
            elif type == 2: # end case
                pass
            elif type == 3: # timeout
                pass
            else: # error
                log("relaySecondModel", "Wrong type recieved")

            print("[RECEIVED]", row)
        except Empty:
            time.sleep(0.5)

if __name__ == "__main__":
    from threading import Thread
    Thread(target=start_queue_server, daemon=True).start()
    consume_loop()