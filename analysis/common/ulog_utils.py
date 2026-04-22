#!/usr/bin/env python3
"""Shared ULog parsing utilities for analysis scripts."""

import numpy as np

try:
    from pyulog import ULog
except ImportError:
    ULog = None

from analysis.common.geo import latlon_to_meters


def load_ulog(path):
    """Load a ULog file and return the ULog object."""
    if ULog is None:
        raise ImportError("pyulog is required: pip install pyulog")
    return ULog(str(path))


def list_topics(ulog):
    """Return a list of topic names present in *ulog*."""
    return [d.name for d in ulog.data_list]


def extract_gps_trajectory(ulog, ref_lat=None, ref_lon=None):
    """Extract GPS trajectory from a ULog file."""
    topics = list_topics(ulog)

    gps_topic = None
    for name in ("vehicle_global_position_groundtruth", "vehicle_global_position"):
        if name in topics:
            gps_topic = ulog.get_dataset(name)
            break

    if gps_topic is None:
        return None

    data = gps_topic.data
    timestamp = np.array(data["timestamp"])
    time = (timestamp - timestamp[0]) / 1e6

    if "lat" not in data or "lon" not in data:
        return None

    lat = np.array(data["lat"])
    lon = np.array(data["lon"])

    result = {"time": time, "lat": lat, "lon": lon}

    if "alt" in data:
        result["alt"] = np.array(data["alt"])

    if ref_lat is not None and ref_lon is not None:
        x, y = latlon_to_meters(lat, lon, ref_lat, ref_lon)
        result["x"] = x
        result["y"] = y

    return result


def extract_local_trajectory(ulog):
    """Extract local NED trajectory from a ULog file."""
    topics = list_topics(ulog)
    if "vehicle_local_position" not in topics:
        return None

    local_pos = ulog.get_dataset("vehicle_local_position")
    data = local_pos.data
    timestamp = np.array(data["timestamp"])
    time = (timestamp - timestamp[0]) / 1e6

    result = {"time": time}
    for key in ("x", "y", "z", "vx", "vy", "vz"):
        if key in data:
            result[key] = np.array(data[key])
    return result


def extract_bias_data(ulog):
    """Extract injected gyroscope bias from a ULog file."""
    topics = list_topics(ulog)
    if "gyro_bias" not in topics:
        return None

    bias_topic = ulog.get_dataset("gyro_bias")
    data = bias_topic.data
    timestamp = np.array(data["timestamp"])
    time = (timestamp - timestamp[0]) / 1e6

    return {
        "time": time,
        "bias_x": np.array(data["gyro_bias_x"]),
        "bias_y": np.array(data["gyro_bias_y"]),
        "bias_z": np.array(data["gyro_bias_z"]),
    }


def extract_ekf_bias(ulog):
    """Extract EKF-estimated gyroscope bias from a ULog file."""
    topics = list_topics(ulog)
    if "estimator_sensor_bias" not in topics:
        return None

    ekf_topic = ulog.get_dataset("estimator_sensor_bias")
    data = ekf_topic.data
    timestamp = np.array(data["timestamp"])
    time = (timestamp - timestamp[0]) / 1e6

    return {
        "time": time,
        "bias_x": np.array(data["gyro_bias[0]"]),
        "bias_y": np.array(data["gyro_bias[1]"]),
        "bias_z": np.array(data["gyro_bias[2]"]),
    }


def detect_attack_period(bias_data, threshold=1e-4):
    """Detect the start and end times of a bias injection attack."""
    if bias_data is None:
        return None, None

    magnitude = np.abs(bias_data["bias_x"]) + np.abs(bias_data["bias_y"]) + np.abs(bias_data["bias_z"])
    active = magnitude > threshold

    if not np.any(active):
        return None, None

    indices = np.where(active)[0]
    start_time = bias_data["time"][indices[0]]
    end_time = bias_data["time"][indices[-1]]
    return start_time, end_time
