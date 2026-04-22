"""Real-drone MAVLink controller for RQ6."""

from __future__ import annotations

import math
import os
import time

import numpy as np

os.environ["MAVLINK20"] = "1"
from pymavlink import mavutil

from evaluation.sim_to_real.common.config import STANDARD_GRAVITY
from evaluation.sim_to_real.real.config import REQUIRED_TYPES, logger


class PX4RealController:
    def __init__(self, connection: str = "udpout:127.0.0.1:17000"):
        self.master = mavutil.mavlink_connection(connection)
        self.current_gyro_bias = np.zeros(3, dtype=np.float32)
        self.prev_gps_alt = None
        self.prev_gps_timestamp = None

        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )
        try:
            self.master.wait_heartbeat(timeout=5.0)
            logger.info(
                "[PX4Real] Heartbeat received: sys=%s, comp=%s",
                self.master.target_system,
                self.master.target_component,
            )
        except Exception as exc:
            logger.warning("[PX4Real] Heartbeat wait failed: %s", exc)

    def drain_all(self):
        while True:
            msg = self.master.recv_match(blocking=False)
            if msg is None:
                return

    def set_gyro_bias(self, bias: np.ndarray):
        self.master.mav.set_gyro_bias_send(
            float(bias[0]),
            float(bias[1]),
            float(bias[2]),
        )
        self.current_gyro_bias = np.asarray(bias, dtype=np.float32)

    def get_observation(self, timeout: float = 1.0):
        latest = {name: None for name in REQUIRED_TYPES}
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            msg = self.master.recv_match(
                type=REQUIRED_TYPES,
                blocking=True,
                timeout=remaining,
            )
            if msg is None:
                break
            latest[msg.get_type()] = msg
            if all(latest.values()):
                break

        if not all(latest.values()):
            return None

        gps_msg = latest["GPS_RAW_INT"]
        att_msg = latest["ATTITUDE_QUATERNION"]
        imu_msg = latest["SCALED_IMU"]

        ground_speed_m_s = gps_msg.vel / 100.0
        cog_rad = math.radians(gps_msg.cog / 100.0)
        vx_gps = ground_speed_m_s * math.cos(cog_rad)
        vy_gps = ground_speed_m_s * math.sin(cog_rad)

        current_alt = gps_msg.alt / 1000.0
        current_timestamp = gps_msg.time_usec / 1e6
        if self.prev_gps_alt is not None and self.prev_gps_timestamp is not None:
            delta_t = current_timestamp - self.prev_gps_timestamp
            vz_gps = (current_alt - self.prev_gps_alt) / delta_t if delta_t > 0 else 0.0
        else:
            vz_gps = 0.0
        self.prev_gps_alt = current_alt
        self.prev_gps_timestamp = current_timestamp

        xacc = imu_msg.xacc * STANDARD_GRAVITY / 1000.0
        yacc = imu_msg.yacc * STANDARD_GRAVITY / 1000.0
        zacc = imu_msg.zacc * STANDARD_GRAVITY / 1000.0

        xgyro = (imu_msg.xgyro / 1000.0) - self.current_gyro_bias[0]
        ygyro = (imu_msg.ygyro / 1000.0) - self.current_gyro_bias[1]
        zgyro = (imu_msg.zgyro / 1000.0) - self.current_gyro_bias[2]

        return np.array([
            gps_msg.lat * 1e-7,
            gps_msg.lon * 1e-7,
            current_alt,
            vx_gps,
            vy_gps,
            vz_gps,
            xacc,
            yacc,
            zacc,
            att_msg.q1,
            att_msg.q2,
            att_msg.q3,
            att_msg.q4,
            xgyro,
            ygyro,
            zgyro,
        ], np.float32)

    def get_hil(self, timeout: float = 1.0):
        return self.get_observation(timeout=timeout)
