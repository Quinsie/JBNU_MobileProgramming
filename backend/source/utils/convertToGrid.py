# backend/source/utils/convertToGrid.py
# 기상청 API를 활용하기 위해 위경도를 격자형태 좌표로 바꿔주는 함수

import math

def convert_to_grid(lat, lon):
    RE = 6371.00877  # Earth radius (km)
    GRID = 5.0       # Grid spacing (km)
    SLAT1 = 30.0     # Projection latitude 1 (degree)
    SLAT2 = 60.0     # Projection latitude 2 (degree)
    OLON = 126.0     # Origin longitude (degree)
    OLAT = 38.0      # Origin latitude (degree)
    XO = 43          # Origin X coordinate (GRID)
    YO = 136         # Origin Y coordinate (GRID)

    DEGRAD = math.pi / 180.0

    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn * math.cos(slat1)) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)

    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    x = ra * math.sin(theta) + XO + 0.5
    y = ro - ra * math.cos(theta) + YO + 0.5

    return int(x), int(y)