"""ArduPilot 3.4 controller for the CI-detector VM path."""

import os
import time
import math
import numpy as np
from typing import Optional, Tuple
import logging

# Legacy AP 3.4 still speaks MAVLink 1.
os.environ.pop("MAVLINK20", None)
os.environ.pop("MAVLINK09", None)
from pymavlink import mavutil

logger = logging.getLogger(__name__)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert Euler angles to a quaternion."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,  # w
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
    )


class ArdupilotLegacyController:
    """AP 3.4 controller with SIM_GYR_BIAS_X/Y bias injection."""

    def __init__(
        self,
        frame_type: str = "quad",
        connection_string: str = "udp:0.0.0.0:17000",
        speedup: int = 1,
        home_location: Tuple[float, float, float, float] = None,
    ):
        """AP 3.4 ignores `speedup` and `home_location`."""
        self.frame_type = frame_type
        self.connection_string = connection_string
        self.speedup = speedup

        logger.info(f"[Ardupilot-Legacy-{frame_type}] Connecting to {connection_string}")

        self.master = mavutil.mavlink_connection(
            connection_string,
            source_system=246,
            source_component=mavutil.mavlink.MAV_COMP_ID_MISSIONPLANNER
        )
        self.master.wait_heartbeat()
        logger.info(
            f"[Ardupilot-Legacy-{frame_type}] sys={self.master.target_system} "
            f"comp={self.master.target_component} mavlink={self.master.WIRE_PROTOCOL_VERSION}"
        )

        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10, 1,
        )
        time.sleep(0.5)

        # AP 3.4 has no GET_GYRO_BIAS reply path.
        self.current_gyro_bias = np.zeros(3, dtype=np.float32)

        self.ci_detector_enabled = False
        self.attack_detected = False
        self.detection_time = None

    def get_state(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read one SIM_STATE sample in the original 16-float layout."""
        msg = self.master.recv_match(type='SIM_STATE', blocking=True, timeout=timeout)
        if not msg:
            logger.warning(f"[Ardupilot-Legacy-{self.frame_type}] get_state timeout")
            return None

        arr = np.array([
            msg.lat, msg.lon, msg.alt,
            msg.vn, msg.ve, msg.vd,
            msg.xacc, msg.yacc, msg.zacc,
            msg.q1, msg.q2, msg.q3, msg.q4,
            msg.xgyro, msg.ygyro, msg.zgyro,
        ], dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def get_gyro_bias(self) -> np.ndarray:
        return self.current_gyro_bias.copy()

    def set_gyro_bias(self, bias: np.ndarray) -> bool:
        """Write bias[0..1] to SIM_GYR_BIAS_X/Y."""
        logger.debug(
            f"[Ardupilot-Legacy-{self.frame_type}] bias="
            f"[{bias[0]:.6f}, {bias[1]:.6f}, {bias[2]:.6f}] rad/s"
        )
        for axis_name, value in (("SIM_GYR_BIAS_X", bias[0]),
                                 ("SIM_GYR_BIAS_Y", bias[1])):
            self.master.mav.param_set_send(
                self.master.target_system,
                mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1,
                axis_name.encode(),
                float(value),
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            )
        self.current_gyro_bias = bias.copy()
        return True

    def check_attack_detection(self, episode_start_time: float = None) -> bool:
        """Latch once STATUSTEXT reports ATTACK DETECTED."""
        while True:
            msg = self.master.recv_match(type='STATUSTEXT', blocking=False)
            if msg is None:
                break
            if "ATTACK DETECTED" in msg.text.upper():
                if not self.attack_detected:
                    self.attack_detected = True
                    self.detection_time = (
                        time.time() - episode_start_time
                        if episode_start_time is not None else time.time()
                    )
                    logger.warning(
                        f"[Ardupilot-Legacy-{self.frame_type}] Attack detected "
                        f"at t={self.detection_time:.2f}s"
                    )
                return True
        return self.attack_detected

    def get_detection_info(self) -> dict:
        return {
            "detected": self.attack_detected,
            "detection_time": self.detection_time,
            "ci_detector_enabled": self.ci_detector_enabled
        }

    def reset_detection_state(self):
        self.attack_detected = False
        self.detection_time = None

    def drain_all(self):
        drained = 0
        while True:
            m = self.master.recv_match(blocking=False)
            if m is None:
                break
            drained += 1
        if drained > 100:
            logger.debug(f"[Ardupilot-Legacy-{self.frame_type}] drained {drained} msgs")

    def close(self):
        if hasattr(self, 'master') and self.master:
            self.master.close()
            self.master = None  # prevent double-close from __del__

    def __del__(self):
        try:
            if hasattr(self, 'master') and self.master:
                self.close()
        except Exception:
            pass
