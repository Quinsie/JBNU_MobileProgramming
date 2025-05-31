# backend/source/ai/generateFirstETA.py

import os
import sys
import json
import time
import torch
import argparse
from datetime import datetime, timedelta
from FirstETAModel import FirstETAModel

# ==== 경로 설정 ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

# ==== 경로 상수 ====
STOPS_DIR = os.path.join(BASE_DIR, "data", "raw", "staticInfo", "stops")
FORECAST_DIR = os.path.join(BASE_DIR, "data", "raw", "dynamicInfo", "forecast")
MEAN_ELAPSED_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "elapsed")
MEAN_INTERVAL_DIR = os.path.join(BASE_DIR, "data", "processed", "mean", "interval")
DEPARTURE_CACHE_DIR = os.path.join(BASE_DIR, "data", "processed", "departure_cache")

LABEL_BUS_PATH = os.path.join(BASE_DIR, "data", "processed", "label_bus.json")
LABEL_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "label_stops.json")
NX_NY_STOP_PATH = os.path.join(BASE_DIR, "data", "processed", "nx_ny_stops.json")
STDID_NUMBER_PATH = os.path.join(BASE_DIR, "data", "processed", "stdid_number.json")

SAVE_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "eta_table", "first_model")
os.makedirs(SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None

# ==== 보조 함수 ====
from source.utils.normalize import normalize
from source.utils.getDayType import getDayType
from source.utils.getTimeGroup import getTimeGroup
from source.utils.timeToSinCos import time_to_sin_cos
from source.utils.fallbackForecast import fallback_forecast
from source.utils.extractRouteInfo import extract_route_info

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def set_global_model(model):
    global MODEL
    MODEL = model
    MODEL.eval()

# ==== 단일 stdid 처리 함수 ====
def infer_single(nx_ny_stops, entry, target_date, wd_label, stdid_number, label_bus, label_stops, mean_elapsed, mean_interval, forecast_all):
    hhmm = entry['time']  # ex: "07:50"
    dep_hour, dep_minute = map(int, hhmm.split(":"))
    save_hour, save_minute = hhmm.split(":")
    dep = datetime(target_date.year, target_date.month, target_date.day, dep_hour, dep_minute)
    result = {}

    global MODEL

    for stdid in entry["stdid"]:
        stop_file = os.path.join(STOPS_DIR, f"{stdid}.json")
        if not os.path.exists(stop_file): continue
        stops = load_json(stop_file)['resultList']
        bn, dr, br = extract_route_info(stdid, stdid_number, label_bus)
        max_ord = max(s["STOP_ORD"] for s in stops)
        eta_dict, prev_elapsed = {}, 0.0

        for s in stops:
            ord = s["STOP_ORD"]; stop_id = s["STOP_ID"]
            eta_time = dep + timedelta(seconds=prev_elapsed)
            if ord == 1:
                eta_dict[str(ord)] = eta_time.strftime("%Y-%m-%d %H:%M:%S")
                continue

            stop_idx = int(label_stops.get(str(stop_id), 0))
            tg = getTimeGroup(dep)
            wd_tg = wd_label * 8 + (tg - 1)
            nx_ny = nx_ny_stops.get(f"{stdid}_{ord}", "63_89")
            weather = fallback_forecast(dep, nx_ny, forecast_all)
            me = mean_elapsed.get(str(stdid), {}).get(str(ord), {})
            mi = mean_interval.get(str(stop_id), {})

            # === 평균값 fallback 구조 적용 ===
            me = mean_elapsed.get(str(stdid), {}).get(str(ord), {})
            mi = mean_interval.get(str(stop_id), {})
            pme = mean_elapsed.get(str(stdid), {}).get(str(ord - 1), {})

            raw_me_total = me.get("total", {}).get("mean", None)
            me_total = normalize(raw_me_total, 0, 7200) if raw_me_total is not None else 0.0

            raw_me_weekday = me.get(f"weekday_{wd_label}", {}).get("mean", None)
            me_weekday = normalize(raw_me_weekday, 0, 7200) if raw_me_weekday is not None else me_total

            raw_me_timegroup = me.get(f"timegroup_{tg}", {}).get("mean", None)
            me_timegroup = normalize(raw_me_timegroup, 0, 7200) if raw_me_timegroup is not None else me_total

            raw_me_wd_tg = me.get(f"wd_tg_{wd_label}_{tg}", {}).get("mean", None)
            if raw_me_wd_tg is not None:
                me_wd_tg = normalize(raw_me_wd_tg, 0, 7200)
            elif raw_me_weekday is not None:
                me_wd_tg = me_weekday
            elif raw_me_timegroup is not None:
                me_wd_tg = me_timegroup
            else:
                me_wd_tg = me_total

            raw_mi_total = mi.get("total", {}).get("mean", None)
            mi_total = normalize(raw_mi_total, 0, 600) if raw_mi_total is not None else 0.0

            raw_mi_weekday = mi.get(f"weekday_{wd_label}", {}).get("mean", None)
            mi_weekday = normalize(raw_mi_weekday, 0, 600) if raw_mi_weekday is not None else mi_total

            raw_mi_timegroup = mi.get(f"timegroup_{tg}", {}).get("mean", None)
            mi_timegroup = normalize(raw_mi_timegroup, 0, 600) if raw_mi_timegroup is not None else mi_total

            raw_mi_wd_tg = mi.get(f"wd_tg_{wd_label}_{tg}", {}).get("mean", None)
            if raw_mi_wd_tg is not None:
                mi_wd_tg = normalize(raw_mi_wd_tg, 0, 600)
            elif raw_mi_weekday is not None:
                mi_wd_tg = mi_weekday
            elif raw_mi_timegroup is not None:
                mi_wd_tg = mi_timegroup
            else:
                mi_wd_tg = mi_total

            raw_pme_total = pme.get("total", {}).get("mean", None)
            pme_total = normalize(raw_pme_total, 0, 7200) if raw_pme_total is not None else 0.0

            raw_pme_weekday = pme.get(f"weekday_{wd_label}", {}).get("mean", None)
            pme_weekday = normalize(raw_pme_weekday, 0, 7200) if raw_pme_weekday is not None else pme_total

            raw_pme_timegroup = pme.get(f"timegroup_{tg}", {}).get("mean", None)
            pme_timegroup = normalize(raw_pme_timegroup, 0, 7200) if raw_pme_timegroup is not None else pme_total

            raw_pme_wd_tg = pme.get(f"wd_tg_{wd_label}_{tg}", {}).get("mean", None)
            if raw_pme_wd_tg is not None:
                pme_wd_tg = normalize(raw_pme_wd_tg, 0, 7200)
            elif raw_pme_weekday is not None:
                pme_wd_tg = pme_weekday
            elif raw_pme_timegroup is not None:
                pme_wd_tg = pme_timegroup
            else:
                pme_wd_tg = pme_total

            row = {
                "x_bus_number": bn,
                "x_direction": dr,
                "x_branch": br,
                "x_weekday": wd_label,
                "x_timegroup": tg,
                "x_weekday_timegroup": wd_tg,
                "x_mean_elapsed_total": me_total,
                "x_mean_elapsed_weekday": me_weekday,
                "x_mean_elapsed_timegroup": me_timegroup,
                "x_mean_elapsed_wd_tg": me_wd_tg,
                "x_prev_mean_elapsed_total": pme_total,
                "x_prev_mean_elapsed_weekday": pme_weekday,
                "x_prev_mean_elapsed_timegroup": pme_timegroup,
                "x_prev_mean_elapsed_wd_tg": pme_wd_tg,
                "x_node_id": stop_idx,
                "x_mean_interval_total": mi_total,
                "x_mean_interval_weekday": mi_weekday,
                "x_mean_interval_timegroup": mi_timegroup,
                "x_mean_interval_wd_tg": mi_wd_tg,
                "x_weather_PTY": weather["PTY"],
                "x_weather_RN1": normalize(weather['RN1'], 0, 100),
                "x_weather_T1H": normalize(weather['T1H'], -30, 50),
                "x_departure_time_sin": time_to_sin_cos(dep)[0],
                "x_departure_time_cos": time_to_sin_cos(dep)[1],
                "x_ord_ratio": round(ord / max_ord, 4),
                "x_prev_ord_ratio": round((ord-1) / max_ord, 4),
                "x_prev_pred_elapsed": 0.0
            }
            # print(bn, dr, br, wd_label, tg, wd_tg, me_total, me_weekday, me_timegroup, me_wd_tg, stop_idx, mi_total, mi_weekday, mi_timegroup, mi_wd_tg, weather, time_to_sin_cos(dep)[0], time_to_sin_cos(dep)[1], round(ord / max_ord, 4))

            int_keys = {
                "bus_number", "direction", "branch", "node_id", "weekday", "timegroup", "weekday_timegroup", "weather_PTY"
            }
            float_keys = {
                "mean_elapsed_total", "mean_elapsed_weekday", "mean_elapsed_timegroup", "mean_elapsed_wd_tg",
                "prev_mean_elapsed_total", "prev_mean_elapsed_weekday", "prev_mean_elapsed_timegroup", "prev_mean_elapsed_wd_tg",
                "mean_interval_total", "mean_interval_weekday", "mean_interval_timegroup", "mean_interval_wd_tg",
                "weather_RN1", "weather_T1H",
                "departure_time_sin", "departure_time_cos",
                "ord_ratio", "prev_pred_elapsed"
            }

            x_tensor = {}
            for k, v in row.items():
                key = k.replace("x_", "")

                if key in float_keys:
                    val = torch.tensor([v], dtype=torch.float32)  # shape: (1, 1)
                elif key in int_keys:
                    val = torch.tensor([v], dtype=torch.long)     # shape: (1, 1)
                
                if val.dim() == 1 and val.dtype == torch.float32:
                    val = val.unsqueeze(1)

                x_tensor[key] = val.to(device)

            with torch.no_grad():
                pred_mean, _ = MODEL(x_tensor)
                elapsed = float(pred_mean.item()) * 7200

                eta_time = dep + timedelta(seconds=elapsed)
                eta_dict[str(ord)] = eta_time.strftime("%Y-%m-%d %H:%M:%S")

        result[f"{stdid}_{save_hour}{save_minute}"] = eta_dict
    return result

# ===== main =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    now = time.time()
    print(f"ETA Table 생성 시작... {args.date}")

    target_date = datetime.strptime(args.date, "%Y%m%d")
    date_str = args.date
    prev_date_str = (target_date - timedelta(days=1)).strftime("%Y%m%d")
    weekday_type = getDayType(target_date)
    wd_label = {"weekday": 1, "saturday": 2, "holiday": 3}[weekday_type]

    stdid_number = load_json(STDID_NUMBER_PATH)
    label_bus = load_json(LABEL_BUS_PATH)
    label_stops = load_json(LABEL_STOP_PATH)
    mean_elapsed = load_json(os.path.join(MEAN_ELAPSED_DIR, f"{prev_date_str}.json"))
    mean_interval = load_json(os.path.join(MEAN_INTERVAL_DIR, f"{prev_date_str}.json"))
    dep_data = load_json(os.path.join(DEPARTURE_CACHE_DIR, f"{weekday_type}.json"))["data"]
    with open(NX_NY_STOP_PATH, encoding='utf-8') as f: nx_ny_stops = json.load(f)

    target_date_str = target_date.strftime("%Y%m%d")
    prev_date_str = (target_date - timedelta(days=1)).strftime("%Y%m%d")
    prev2_date_str = (target_date - timedelta(days=2)).strftime("%Y%m%d")
    valid_prefixes = {target_date_str, prev_date_str, prev2_date_str}
    forecast_all = {
        f.replace(".json", ""): load_json(os.path.join(FORECAST_DIR, f))
        for f in os.listdir(FORECAST_DIR)
        if f.endswith(".json") and any(f.startswith(prefix) for prefix in valid_prefixes)
    }

    model_pth = os.path.join(BASE_DIR, "data", "model", "firstETA", "replay", f"{date_str}.pth")
    model_obj = FirstETAModel(); model_obj.load_state_dict(torch.load(model_pth, map_location=device))
    set_global_model(model_obj.to(device))

    task_args = [(nx_ny_stops, entry, target_date, wd_label, stdid_number, label_bus, label_stops, mean_elapsed, mean_interval, forecast_all) for entry in dep_data]
    
    def unpack_and_infer(args): return infer_single(*args)
    results = [unpack_and_infer(args) for args in task_args]

    final = {}; [final.update(r) for r in results]
    with open(os.path.join(SAVE_PATH, f"{date_str}.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    print(f"[INFO] ETA 저장 완료: {date_str} / 총 {len(final)}개 운행")
    print("소요 시간: ", time.time() - now, "sec")