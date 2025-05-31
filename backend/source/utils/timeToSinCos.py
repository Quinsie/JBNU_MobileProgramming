# backend/source/utils/timeToSinCos.py

import math
import typing
import datetime

def time_to_sin_cos(dt: datetime.datetime) -> typing.Tuple[float, float]:
    minutes = dt.hour * 60 + dt.minute
    angle = 2 * math.pi * (minutes / 1440)
    return math.sin(angle), math.cos(angle)