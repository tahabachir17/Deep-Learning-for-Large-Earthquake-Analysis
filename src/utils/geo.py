"""Geodetic helpers used across training and inference."""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0
KM_PER_DEGREE = 111.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def km_to_deg(distance_km: float) -> float:
    return distance_km / KM_PER_DEGREE


def azimuth_deg(event_lat: float, event_lon: float, station_lat: float, station_lon: float) -> float:
    lat1 = math.radians(event_lat)
    lat2 = math.radians(station_lat)
    dlon = math.radians(station_lon - event_lon)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(x, y)) % 360.0
