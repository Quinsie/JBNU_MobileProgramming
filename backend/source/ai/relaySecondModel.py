# backend/source/ai/relaySecondModel.py

from queue import Empty
from multiprocessing import Queue
from multiprocessing.managers import BaseManager
import time

# 실행날짜 기억하고, json파일로 저장은 "당일 날짜"로 진행해야된다.
# 절대 잊지 말 것!!!
# 중계기는 실행과 동시에 해당 날짜의 first eta 예측파일을 second eta 파일로 복사하고, 그걸 계속 수정해야한다.
# 또한 추론을 위한 json 저장은 별도로 진행한다. 얘도 한 파일에 넣고.

# === Queue 생성 ===
q = Queue()

# === Queue 공유할 Manager 설정 ===
class QueueManager(BaseManager): pass
QueueManager.register("get_queue", callable=lambda: q)

def start_queue_server():
    manager = QueueManager(address=("localhost", 50000), authkey=b"abc")
    server = manager.get_server()
    print("[RELAY] Queue 서버 시작됨 (포트 50000)")
    server.serve_forever()

def consume_loop():
    print("[RELAY] Queue 소비 시작")
    while True:
        try:
            row = q.get(timeout = 1)
            print("[RECEIVED]", row)
        except Empty:
            time.sleep(0.5)

if __name__ == "__main__":
    from threading import Thread
    # 하나는 queue server, 하나는 consumer
    Thread(target=start_queue_server, daemon=True).start()
    consume_loop()