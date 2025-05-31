# backend/source/utils/fallbackForecast.py

from datetime import timedelta

def fallback_forecast(target_dt, nx_ny, forecast_all):
    target_keys = [(target_dt - timedelta(hours=h)).strftime('%Y%m%d_%H00') for h in range(0, 4)]
    fallback_dates = [
        target_dt.strftime('%Y%m%d'),
        (target_dt - timedelta(days=1)).strftime('%Y%m%d'),
        (target_dt - timedelta(days=2)).strftime('%Y%m%d')
    ]
    for date_key in fallback_dates:
        forecast = forecast_all.get(date_key)
        if not forecast:
            continue
        for ts in target_keys:
            if ts in forecast:
                nx, ny = map(int, nx_ny.split('_'))
                for dx in [0, -1, 1]:
                    for dy in [0, -1, 1]:
                        key2 = f"{nx + dx}_{ny + dy}"
                        data = forecast[ts].get(key2)
                        if (data and all(k in data for k in ('TMP', 'PCP', 'PTY')) and
                            data['PTY'] is not None and int(data['PTY']) >= 0 and
                            data['PCP'] is not None and int(data['PCP'] >= 0)):
                            return {
                                'T1H': data['TMP'],
                                'RN1': data['PCP'],
                                'PTY': int(data['PTY'])
                            }
    return {'T1H': 20.0, 'RN1': 0.0, 'PTY': 0}