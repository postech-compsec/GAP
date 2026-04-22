"""RQ6 ULog helpers for the logged equivalents of the runtime observation path."""

from __future__ import annotations


def _ds(ulog, name):
    for data in ulog.data_list:
        if data.name == name:
            return data.data
    return None


def select_logged_topics(ulog):
    """Return logged equivalents of GPS_RAW_INT, ATTITUDE_QUATERNION,
    SCALED_IMU, and gyro_bias."""
    gps = _ds(ulog, "sensor_gps")
    attitude = _ds(ulog, "vehicle_attitude")
    imu = _ds(ulog, "sensor_combined")
    bias = _ds(ulog, "gyro_bias")
    return {
        "gps": gps,
        "attitude": attitude,
        "imu": imu,
        "bias": bias,
    }
