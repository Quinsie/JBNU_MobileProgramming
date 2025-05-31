# backend/source/utils/fallbackWeather.py

import datetime

def fallback_weather(target_time: datetime.datetime, nx_ny: str, weather_all: dict):
    tried_timestamps = sorted(weather_all.keys(), reverse=True)

    for i in range(181):  # 최대 3-hour fallback까지 시도
        closest_key = None
        for ts in tried_timestamps:
            ts_dt = datetime.datetime.strptime(ts, "%Y%m%d_%H%M")
            if ts_dt <= target_time:
                closest_key = ts
                break

        grid_data = weather_all[closest_key]
        # nx_ny가 없거나 값이 None이면 fallback
        if nx_ny in grid_data and grid_data[nx_ny] is not None:
            val = grid_data[nx_ny]
            if ( val['PTY'] is not None and val['PTY'] >= 0 and
                val['RN1'] is not None and val['RN1'] >= 0 and
                val['T1H'] is not None ):
                return val

        # 인접 격자 탐색
        nx, ny = map(int, nx_ny.split("_"))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                key2 = f"{nx+dx}_{ny+dy}"
                if key2 in grid_data and grid_data[key2] is not None:
                    val2 = grid_data[key2]
                    if ( val2['PTY'] is not None and val2['PTY'] >= 0 and
                        val2['RN1'] is not None and val2['RN1'] >= 0 and
                        val2['T1H'] is not None ):
                        return val2

        target_time -= datetime.timedelta(minutes=1)

    return {"PTY": 0, "RN1": 0.0, "T1H": 20.0}  # fallback 실패 시 기본값