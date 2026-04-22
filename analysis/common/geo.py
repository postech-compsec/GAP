"""Coordinate utility functions for GPS/geodetic conversions."""

import math
import numpy as np

METERS_PER_DEGREE_LAT = 111319.5

# Derived constant for the project's typical latitude (~47 deg N)
METERS_PER_DEGREE_LON_AT_47 = METERS_PER_DEGREE_LAT * math.cos(math.radians(47.0))


def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    """Convert lat/lon to meters relative to a reference point."""
    earth_radius_m = 6371000.0
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    x = earth_radius_m * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    y = earth_radius_m * (lat_rad - ref_lat_rad)
    return x, y


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate 2D distance in meters between two GPS coordinates."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_m = 6371000
    return c * earth_radius_m
