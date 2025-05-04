# backend/source/scripts/mapCongestion.py

import os
import sys
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from source.utils.haversine import haversine_distance

ROUTE_NODE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_mapped")
TRAFFIC_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "traffic")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed", "route_nodes_cong")
os.makedirs(SAVE_DIR, exist_ok=True)

def load_latest_traffic_file():
    files = sorted([f for f in os.listdir(TRAFFIC_DIR) if f.endswith(".json")])
    if not files:
        raise FileNotFoundError("교통 파일이 존재하지 않습니다.")
    return files[-1]  # 파일명만 반환

def build_traffic_dict(traffic_data):
    traffic_dict = {}
    for item in traffic_data:
        key = (item["id"], item["sub"])
        traffic_dict[key] = item["grade"]
    return traffic_dict

def process_std(stdid, traffic_dict):
    input_path = os.path.join(ROUTE_NODE_DIR, f"{stdid}.json")
    try:
        with open(input_path, encoding="utf-8") as f:
            nodes = json.load(f)["resultList"]
    except Exception as e:
        return stdid, {}  # 실패한 경우 빈 딕셔너리라도 반환

    result = {}
    for i, node in enumerate(nodes):
        matched = node.get("matched")
        if matched is None:
            grade = "1"
        else:
            key = (matched.get("id"), matched.get("sub"))
            grade = traffic_dict.get(key, "1")
        result[str(i)] = {"grade": grade}
    return stdid, result

def main():
    latest_filename = load_latest_traffic_file()
    traffic_path = os.path.join(TRAFFIC_DIR, latest_filename)
    with open(traffic_path, encoding="utf-8") as f:
        traffic_data = json.load(f)
    traffic_dict = build_traffic_dict(traffic_data)

    files = [f for f in os.listdir(ROUTE_NODE_DIR) if f.endswith(".json")]
    stdids = [f.replace(".json", "") for f in files]
    args = [(stdid, traffic_dict) for stdid in stdids]

    result_dict = {}
    with Pool(cpu_count()) as pool:
        for stdid, result in tqdm(pool.starmap(process_std, args), total=len(args)):
            result_dict[stdid] = result

    output_path = os.path.join(SAVE_DIR, latest_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()