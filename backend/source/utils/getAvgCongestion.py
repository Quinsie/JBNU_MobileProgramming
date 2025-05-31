# backend/source/utils/getAvgCongestion.py

import os
import json
from typing import List

def getSingleAvgCongestion(stdid: str, start_node: int, target_node: int, dt_str: str, base_dir: str) -> float:
    try:
        file_path = os.path.join(base_dir, f"{dt_str}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        route_data = data.get(stdid, {})
        if not route_data:
            return 1.0

        if start_node > target_node:
            start_node, target_node = target_node, start_node

        grades = []
        for nid in range(start_node, target_node + 1):
            info = route_data.get(str(nid), {})
            grade = info.get("grade", None)
            if grade in ("1", "2", "3"):
                grades.append(int(grade))

        return sum(grades) / len(grades) if grades else 1.0

    except Exception:
        return 1.0

def getMultiAvgCongestion(stdid: str, start_node: int, target_nodes: List[int], dt_str: str, base_dir: str) -> List[float]:
    try:
        file_path = os.path.join(base_dir, f"{dt_str}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        route_data = data.get(stdid, {})
        if not route_data:
            return [1.0 for _ in target_nodes]

        # start_node ~ max(target_node)까지 미리 누적 계산
        max_target = max(target_nodes)
        if start_node > max_target:
            start_node, max_target = max_target, start_node

        grades = []
        for nid in range(start_node, max_target + 1):
            info = route_data.get(str(nid), {})
            grade = info.get("grade", None)
            if grade in ("1", "2", "3"):
                grades.append(int(grade))
            else:
                grades.append(1)

        # 누적합 구성
        cum_sum = [0]
        for g in grades:
            cum_sum.append(cum_sum[-1] + g)

        result = []
        for target_node in target_nodes:
            a, b = sorted((start_node, target_node))
            left = a - start_node
            right = b - start_node + 1
            length = right - left
            if length <= 0:
                result.append(1.0)
            else:
                avg = (cum_sum[right] - cum_sum[left]) / length
                result.append(avg)

        return result

    except Exception:
        return [1.0 for _ in target_nodes]