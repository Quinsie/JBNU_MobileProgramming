# backend/source/utils/extractRouteInfo.py

def extract_route_info(stdid, stdid_number, label_bus):
    route_str = stdid_number[str(stdid)]
    bus_number_str = route_str[:-2]
    bus_number = int(label_bus.get(bus_number_str, 0))
    direction = 0 if route_str[-2] == 'A' else 1
    branch = int(route_str[-1])
    return bus_number, direction, branch