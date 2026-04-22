"""
Utility functions for the testing framework.

Extracted from test_config.py so constants stay declarative and
functions live with the other helpers.
"""
import math

import numpy as np
from geographiclib.geodesic import Geodesic

from .test_config import (
    MAX_GYRO_BIAS,
    METERS_PER_DEGREE_LAT,
    SUCCESS_CRITERIA,
    TARGET_LAT,
    TARGET_LON,
)


def get_start_positions(num_positions: int, radius: float):
    """
    Generate start positions evenly distributed around a circle.
    Positions are at clock positions (12, 1, 2, 3, ..., 11).
    Used for baseline tests where drone starts around the target.

    Args:
        num_positions: Number of positions to generate.
        radius: Radius of the circle in meters.

    Returns:
        List of dicts with 'lat', 'lon', 'position_index', 'clock_position' keys.
    """
    positions = []
    for i in range(num_positions):
        theta = 2 * math.pi * i / num_positions
        offset_north = radius * math.cos(theta)
        offset_east = radius * math.sin(theta)

        dlat = offset_north / METERS_PER_DEGREE_LAT
        dlon = offset_east / (METERS_PER_DEGREE_LAT * math.cos(math.radians(TARGET_LAT)))

        positions.append({
            "lat": TARGET_LAT + dlat,
            "lon": TARGET_LON + dlon,
            "position_index": i,
            "clock_position": i if i > 0 else 12,
        })

    return positions


def get_target_positions(drone_lat: float, drone_lon: float, num_positions: int, radius: float):
    """
    Generate target positions evenly distributed around a fixed drone position.
    Targets are at clock positions (12, 1, 2, 3, ..., 11).
    Used for RL tests where drone is at center and targets move around it.

    Args:
        drone_lat: Drone's fixed latitude (center position).
        drone_lon: Drone's fixed longitude (center position).
        num_positions: Number of target positions to generate.
        radius: Radius of the circle in meters.

    Returns:
        List of dicts with 'target_lat', 'target_lon', 'drone_lat', 'drone_lon',
        'position_index', 'clock_position' keys.
    """
    positions = []
    for i in range(num_positions):
        theta = 2 * math.pi * i / num_positions
        offset_north = radius * math.cos(theta)
        offset_east = radius * math.sin(theta)

        dlat = offset_north / METERS_PER_DEGREE_LAT
        dlon = offset_east / (METERS_PER_DEGREE_LAT * math.cos(math.radians(drone_lat)))

        positions.append({
            "target_lat": drone_lat + dlat,
            "target_lon": drone_lon + dlon,
            "drone_lat": drone_lat,
            "drone_lon": drone_lon,
            "position_index": i,
            "clock_position": i if i > 0 else 12,
        })

    return positions


def calculate_directional_attack(position_index: int, num_positions: int = 12):
    """
    Calculate gyro attack values based on drone position around the circle.
    Attack is designed to make the drone move toward the center.

    For directional attack:
    - At 12 o'clock (North, theta=0): pitch forward -> move south (toward center)
    - At 3 o'clock (East, theta=pi/2): roll right -> move west (toward center)
    - At 6 o'clock (South, theta=pi): pitch backward -> move north (toward center)
    - At 9 o'clock (West, theta=3pi/2): roll left -> move east (toward center)

    Args:
        position_index: Index of position (0-11, where 0 is 12 o'clock).
        num_positions: Total number of positions (default 12).

    Returns:
        numpy array [x_bias, y_bias, z_bias] in rad/s.
    """
    theta = 2 * math.pi * position_index / num_positions
    magnitude = MAX_GYRO_BIAS
    x_bias = magnitude * math.sin(theta)    # East: positive, West: negative
    y_bias = -magnitude * math.cos(theta)   # North: negative, South: positive
    return np.array([x_bias, y_bias, 0.0], dtype=np.float32)


def check_success(current_lat, current_lon, current_alt,
                  target_lat, target_lon, target_alt,
                  success_type="sphere_10m"):
    """
    Check if drone has reached the success zone.

    Args:
        current_lat, current_lon, current_alt: Current drone position.
        target_lat, target_lon, target_alt: Target position.
        success_type: Key from SUCCESS_CRITERIA dict.

    Returns:
        bool: True if in success zone.
    """
    criteria = SUCCESS_CRITERIA[success_type]
    radius = criteria["radius"]

    geodesic = Geodesic.WGS84.Inverse(current_lat, current_lon, target_lat, target_lon)
    horizontal_distance = geodesic['s12']

    if criteria["type"] == "sphere":
        vertical_distance = current_alt - target_alt
        distance_3d = np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2)
        return distance_3d <= radius
    if criteria["type"] == "cylinder":
        return horizontal_distance <= radius
    return False
