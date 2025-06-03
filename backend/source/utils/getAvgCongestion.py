# backend/source/utils/getAvgCongestion.py

import datetime

def get_avg_congestion_list(now_ord, max_ord, now_node, route_nodes_pair, route_nodes_mapped, traffic_all, arr_time, stdid):
    ord_node_id_list = [0 for _ in range(5)]
    avg_congestion_list = [0.0 for _ in range(5)]
    length = 0

    for i in range(1, 6):
        target_ord = now_ord + i
        if target_ord > max_ord: break
        ord_node_id_list[i - 1] = route_nodes_pair[stdid][str(target_ord)][0]

    length = len([x for x in ord_node_id_list if x != 0])

    total = 0.0; num = 0; now_idx = 0
    for i in range(now_node, max(ord_node_id_list) + 1):
        if i == ord_node_id_list[now_idx]:
            avg_congestion_list[now_idx] = total / num if num > 0 else 0.0
            now_idx += 1
            if now_idx >= length:
                break

        if not route_nodes_mapped[stdid][i].get("matched"):
            total += 1; num += 1
            continue

        now_id = route_nodes_mapped[stdid][i]["matched"]["id"]
        now_sub = route_nodes_mapped[stdid][i]["matched"]["sub"]
        now_grade = 1
        for j in range(5):
            time_key = (arr_time - datetime.timedelta(minutes=j)).strftime("%Y%m%d_%H%M")
            if time_key in traffic_all:
                record = traffic_all[time_key].get((now_id, now_sub))
                if record is not None and 1 <= int(record) <= 3:
                    now_grade = int(record)
                    break
        total += now_grade; num += 1

    return avg_congestion_list