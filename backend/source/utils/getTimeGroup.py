# source/utils/getTimeGroup.py

import datetime

def getTimeGroup(departure_time: datetime.datetime) -> int:
    minutes = departure_time.hour * 60 + departure_time.minute
    if 330 <= minutes < 420: return 1
    elif 420 <= minutes < 540: return 2
    elif 540 <= minutes < 690: return 3
    elif 690 <= minutes < 840: return 4
    elif 840 <= minutes < 1020: return 5
    elif 1020 <= minutes < 1140: return 6
    elif 1140 <= minutes < 1260: return 7
    else: return 8