# backend/source/utils/getDayType.py

import holidays
import datetime

kr_holidays = holidays.KR()

def getDayType(date: datetime.datetime) -> str:
    if date in kr_holidays: # 공휴일
        return "holiday"
    weekday = date.weekday()
    if weekday == 5: # 토요일
        return "saturday"
    elif weekday == 6: # 일요일
        return "holiday"
    return "weekday" # 평일