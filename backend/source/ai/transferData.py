# backend/source/ai/transferData.py

from queue import Empty
from multiprocessing import Queue
from multiprocessing.managers import BaseManager
import time

q = Queue()

# === Queue 공유할 Manager 설정 ===
class QueueManager(BaseManager): pass
QueueManager.register("get_queue", callable=lambda: q)

def start_queue_server():
    manager = QueueManager(address=("localhost", 49000), authkey=b"def")
    server = manager.get_server()
    print("[RELAY] Queue 서버 시작됨 (포트 49000)")
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