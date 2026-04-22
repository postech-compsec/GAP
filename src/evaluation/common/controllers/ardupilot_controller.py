"""ArduPilot SITL controller used by GAP evaluation."""

import os
import time
import subprocess
import math
import signal
import shutil
import glob
import numpy as np

os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil

from ..test_config import STANDARD_GRAVITY, ARDUPILOT_ROOT, logger, SIMULATION_SPEEDUP, LOG_DIR


class ArdupilotController:
    """ArduPilot SITL lifecycle and MAVLink helper."""

    def __init__(self, frame_type: str = "quad", speed_factor: int = None, headless: bool = True):
        """Create one controller for one ArduPilot frame."""
        self.frame_type = frame_type
        self.speed_factor = speed_factor if speed_factor is not None else SIMULATION_SPEEDUP
        self.headless = headless

        self.url = 'udpin:0.0.0.0:17000'
        self.master = mavutil.mavlink_connection(self.url)

        self.mav_recv_timeout = 48 / self.speed_factor
        self.mav_short_timeout = 5 / self.speed_factor

        logger.debug(f"[Ardupilot-{frame_type}] MAVLink connection initialized on {self.url}")

        self.proc = None
        self.home_coordinates = {}

        self.failsafe_events = []
        self.failsafe_keywords = [
            'failsafe', 'Failsafe', 'FAILSAFE',
            'EKF variance', 'EKF primary changed',
            'battery low', 'battery critical', 'Battery',
            'GPS Glitch', 'GPS',
            'RTL', 'LAND', 'land',
            'crash', 'Crash', 'CRASH',
            'disarm', 'Disarm', 'DISARM',
            'Vibration',
            'fence', 'Fence',
            'Throttle', 'RC',
        ]

    def _record_failsafe_text(self, text: str):
        """Record important STATUSTEXT messages.

        We always surface the gyro-bias acceptance text because it is the
        positive audit trail for the attack path. Failsafe-like messages are
        still tracked separately and emitted as warnings.
        """
        text = text.strip()
        if text.startswith("INS: External gyro bias set:"):
            logger.info(f"[Ardupilot-{self.frame_type}] {text}")
            return None

        text_lower = text.lower()
        for keyword in self.failsafe_keywords:
            if keyword.lower() in text_lower:
                event = (time.time(), text)
                self.failsafe_events.append(event)
                logger.warning(f"[Ardupilot-{self.frame_type}] FAILSAFE detected: {text}")
                return event
        return None

    def _request_message(self, message_id: int):
        """Poll one MAVLink message via MAV_CMD_REQUEST_MESSAGE."""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
            0,
            float(message_id),
            0, 0, 0, 0, 0, 0,
        )

    def start(self, latitude: float = None, longitude: float = None):
        """Start one ArduPilot SITL instance."""
        if latitude is not None and longitude is not None:
            self.home_coordinates = {
                'lat': latitude,
                'lon': longitude,
                'alt': 584.0,
            }

        try:
            subprocess.run(["pkill", "-9", "-f", "arducopter"],
                          capture_output=True, timeout=3)
            subprocess.run(["pkill", "-9", "-f", "sim_vehicle.py"],
                          capture_output=True, timeout=3)
            time.sleep(0.5)
            logger.debug(f"[Ardupilot-{self.frame_type}] Cleaned up stale processes")
        except Exception as e:
            logger.debug(f"[Ardupilot-{self.frame_type}] Cleanup error (non-fatal): {e}")

        arducopter_dir = os.path.join(ARDUPILOT_ROOT, "ArduCopter")

        cmd = [
            "python3",
            os.path.join(ARDUPILOT_ROOT, "Tools/autotest/sim_vehicle.py"),
            "-f", self.frame_type,
            "--speedup", str(self.speed_factor),
            "-w",
            "--out", "udpout:127.0.0.1:17000",
        ]

        if not self.headless:
            cmd.extend(["--console", "--map"])

        if latitude is not None and longitude is not None:
            home_str = f"{latitude},{longitude},584,353"
            cmd.extend(["-l", home_str])
            logger.debug(f"[Ardupilot-{self.frame_type}] Start location: {home_str}")

        logger.info(f"[Ardupilot-{self.frame_type}] Starting SITL with speed factor {self.speed_factor}")

        self.proc = subprocess.Popen(
            cmd,
            cwd=arducopter_dir,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL if self.headless else None,
            stderr=subprocess.DEVNULL if self.headless else None
        )

        self.home_coordinates = {}
        logger.info(f"[Ardupilot-{self.frame_type}] Launched SITL process PID={self.proc.pid}")

        wait_time = max(10, 10 / self.speed_factor)
        logger.debug(f"[Ardupilot-{self.frame_type}] Waiting {wait_time:.1f}s for MAVProxy to start...")
        time.sleep(wait_time)

        try:
            self.master.close()
        except:
            pass

        self.master = mavutil.mavlink_connection(self.url)
        logger.debug(f"[Ardupilot-{self.frame_type}] Re-established MAVLink connection")

        return self.proc

    def stop(self):
        """Stop Ardupilot SITL simulator."""
        if self.proc and self.proc.poll() is None:
            pid = self.proc.pid
            pgid = os.getpgid(pid)
            logger.info(f"[Ardupilot-{self.frame_type}] Stopping process group PGID={pgid}")

            # Graceful shutdown
            try:
                os.killpg(pgid, signal.SIGINT)
                self.proc.wait(timeout=10)
                logger.info(f"[Ardupilot-{self.frame_type}] Process stopped gracefully")
                self.proc = None
                return
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass

            # Force kill
            logger.warning(f"[Ardupilot-{self.frame_type}] Graceful shutdown failed. Killing.")
            try:
                os.killpg(pgid, signal.SIGKILL)
                subprocess.run(["pkill", "-9", "-f", "arducopter"], capture_output=True, timeout=3)
            except ProcessLookupError:
                pass

            # Wait for termination
            t0 = time.time()
            while time.time() - t0 < 10:
                if self.proc.poll() is not None:
                    logger.info(f"[Ardupilot-{self.frame_type}] Process confirmed terminated")
                    self.proc = None
                    return
                time.sleep(0.1)

            logger.error(f"[Ardupilot-{self.frame_type}] Failed to confirm termination")

        self.proc = None

    def wait_ready(self, timeout: float = 60):
        """Wait for the first heartbeat."""
        logger.debug(f"[Ardupilot-{self.frame_type}] Waiting for heartbeat...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=self.mav_recv_timeout)
            if msg is not None:
                logger.info(f"[Ardupilot-{self.frame_type}] Heartbeat received")
                return True
        logger.error(f"[Ardupilot-{self.frame_type}] Timeout waiting for heartbeat")
        return False

    def wait_health_ok(self, timeout: float = 120):
        """Wait until position and GPS are available."""
        logger.info(f"[Ardupilot-{self.frame_type}] Requesting data streams...")
        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            10,
            1,
        )
        time.sleep(0.5)

        got_global = got_gps = False

        logger.info(f"[Ardupilot-{self.frame_type}] Waiting for health OK (timeout={timeout}s)...")
        t0 = time.monotonic()
        last_status_time = t0

        while time.monotonic() - t0 < timeout:
            msg = self.master.recv_match(
                type=['GLOBAL_POSITION_INT', 'HOME_POSITION', 'GPS_RAW_INT'],
                blocking=True, timeout=self.mav_recv_timeout
            )

            if time.monotonic() - last_status_time > 5:
                logger.debug(f"[Ardupilot-{self.frame_type}] Health check: global={got_global}, gps={got_gps}")
                last_status_time = time.monotonic()

            if not msg:
                continue
            t = msg.get_type()

            if t == 'GLOBAL_POSITION_INT':
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                alt = msg.alt / 1000.0
                if lat != 0.0 or lon != 0.0:
                    if not got_global:
                        logger.info(f"[Ardupilot-{self.frame_type}] Got GLOBAL_POSITION_INT")
                    got_global = True
                    if not self.home_coordinates:
                        self.home_coordinates = {'lat': lat, 'lon': lon, 'alt': alt}
            elif t == 'HOME_POSITION':
                self.home_coordinates = {
                    'lat': msg.latitude / 1e7,
                    'lon': msg.longitude / 1e7,
                    'alt': msg.altitude / 1000.0
                }
                logger.info(f"[Ardupilot-{self.frame_type}] Got HOME_POSITION: {self.home_coordinates}")
            elif t == 'GPS_RAW_INT':
                if msg.fix_type >= 3:
                    if not got_gps:
                        logger.info(f"[Ardupilot-{self.frame_type}] Got GPS fix (fix_type={msg.fix_type})")
                    got_gps = True

            if got_global and got_gps and self.home_coordinates:
                logger.info(f"[Ardupilot-{self.frame_type}] Health OK - all checks passed. Home: {self.home_coordinates}")
                return True, self.home_coordinates

        logger.error(f"[Ardupilot-{self.frame_type}] Health check timeout: global={got_global}, gps={got_gps}, has_home={bool(self.home_coordinates)}")
        return False, self.home_coordinates

    def set_param(self, param_name: str, param_value: float, timeout: float = 10):
        """Set a parameter value."""
        logger.info(f"[Ardupilot-{self.frame_type}] Setting parameter {param_name}={param_value}")
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            param_name.encode('utf-8'),
            param_value,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )

        # Wait for acknowledgment
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            msg = self.master.recv_match(type='PARAM_VALUE', blocking=True, timeout=1.0)
            if msg and msg.param_id == param_name:
                logger.info(f"[Ardupilot-{self.frame_type}] Parameter {param_name} set to {msg.param_value}")
                return True
        logger.warning(f"[Ardupilot-{self.frame_type}] Failed to confirm parameter {param_name}")
        return False

    def arm_and_takeoff(self, target_alt_m: float, hover_time: float = 60, max_retries: int = 3):
        """Arm and climb to the requested relative altitude."""
        logger.info(f"[Ardupilot-{self.frame_type}] Arm and takeoff to {target_alt_m:.2f}m")

        # Keep EKF events observable instead of forcing an auto-action.
        self.set_param("FS_EKF_ACTION", 0)

        for attempt in range(1, max_retries + 1):
            time.sleep(1 / self.speed_factor)

            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4
            )
            time.sleep(2 / self.speed_factor)

            self.drain_all()

            logger.info(f"[Ardupilot-{self.frame_type}] Sending ARM command (attempt {attempt}/{max_retries})")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )

            arm_ack = None
            status_messages = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.mav_recv_timeout:
                msg = self.master.recv_match(type=['COMMAND_ACK', 'STATUSTEXT'], blocking=True, timeout=1.0)
                if msg:
                    if msg.get_type() == 'COMMAND_ACK' and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                        arm_ack = msg
                        break
                    elif msg.get_type() == 'STATUSTEXT':
                        status_messages.append(msg.text)

            if not (arm_ack and arm_ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
                logger.warning(f"[Ardupilot-{self.frame_type}] Arming failed: {arm_ack.result if arm_ack else 'No ACK'}")
                for status in status_messages:
                    logger.warning(f"[Ardupilot-{self.frame_type}] Status: {status}")
                continue

            logger.info(f"[Ardupilot-{self.frame_type}] Vehicle armed")
            time.sleep(2 / self.speed_factor)

            self.drain_all()

            logger.info(f"[Ardupilot-{self.frame_type}] Sending TAKEOFF command")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, math.nan, 0, 0, target_alt_m
            )

            takeoff_ack = None
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.mav_recv_timeout:
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1.0)
                if msg and msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                    takeoff_ack = msg
                    break

            if not (takeoff_ack and takeoff_ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
                logger.warning(f"[Ardupilot-{self.frame_type}] Takeoff failed, retrying...")
                continue

            logger.info(f"[Ardupilot-{self.frame_type}] Arm and takeoff commands accepted")
            break
        else:
            logger.error(f"[Ardupilot-{self.frame_type}] Failed after {max_retries} retries")
            raise RuntimeError("Failed to arm and takeoff")
        
        self.drain_all()
        logger.info(f"[Ardupilot-{self.frame_type}] Waiting for relative altitude {target_alt_m:.2f}m")
        t0 = time.monotonic()
        while time.monotonic() - t0 < hover_time:
            msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=self.mav_recv_timeout)
            if msg:
                current_altitude_amsl = msg.alt / 1000.0
                if hasattr(msg, "relative_alt"):
                    current_altitude_rel = msg.relative_alt / 1000.0
                elif self.home_coordinates:
                    current_altitude_rel = current_altitude_amsl - self.home_coordinates["alt"]
                else:
                    current_altitude_rel = current_altitude_amsl

                if current_altitude_rel >= target_alt_m - 1:
                    logger.info(
                        f"[Ardupilot-{self.frame_type}] Target altitude reached: "
                        f"rel={current_altitude_rel:.2f}m, AMSL={current_altitude_amsl:.2f}m"
                    )
                    return True

        logger.error(f"[Ardupilot-{self.frame_type}] Failed to reach target altitude")
        return False

    def get_sim_state(self, timeout: float = 1.0):
        """Return one SIM_STATE sample while still recording STATUSTEXT failsafes."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            msg = self.master.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == 'STATUSTEXT':
                self._record_failsafe_text(msg.text if hasattr(msg, 'text') else str(msg))

            if msg_type == 'SIM_STATE':
                arr = np.array([
                    msg.lat_int * 1e-7, msg.lon_int * 1e-7, msg.alt,
                    msg.vn, msg.ve, msg.vd,
                    msg.xacc, msg.yacc, msg.zacc,
                    msg.q1, msg.q2, msg.q3, msg.q4,
                    msg.xgyro, msg.ygyro, msg.zgyro,
                ], np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                return arr

        return None

    def set_gyro_bias(self, bias: np.ndarray):
        """Send the external gyro bias."""
        self.master.mav.set_gyro_bias_send(bias[0], bias[1], bias[2])

    def get_gyro_bias(self, timeout: float = 1.0):
        """Poll the current external gyro bias."""
        self._request_message(mavutil.mavlink.MAVLINK_MSG_ID_GET_GYRO_BIAS)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            wait_time = min(0.1, max(0.0, deadline - time.monotonic()))
            msg = self.master.recv_match(blocking=True, timeout=wait_time)
            if msg is None:
                continue

            msg_type = msg.get_type()

            if msg_type == 'STATUSTEXT':
                self._record_failsafe_text(msg.text if hasattr(msg, 'text') else str(msg))
                continue

            if msg_type == 'COMMAND_ACK':
                if (
                    msg.command == mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE
                    and msg.result != mavutil.mavlink.MAV_RESULT_ACCEPTED
                ):
                    logger.warning(
                        f"[Ardupilot-{self.frame_type}] GET_GYRO_BIAS request denied: result={msg.result}"
                    )
                    return None
                continue

            if msg_type != 'GET_GYRO_BIAS':
                continue

            arr = np.array([msg.gyro_bias_x, msg.gyro_bias_y, msg.gyro_bias_z], np.float32)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr

        return None

    def drain_all(self):
        """Drain buffered MAVLink messages and keep tracking failsafe text."""
        drained = 0
        while True:
            m = self.master.recv_match(blocking=False)
            if m is None:
                break
            drained += 1

            if m.get_type() == 'STATUSTEXT':
                self._record_failsafe_text(m.text if hasattr(m, 'text') else str(m))

        if drained > 100:
            logger.debug(f"[Ardupilot-{self.frame_type}] Drained {drained} messages")

        return drained

    def copy_flight_log(self, position_idx: int, trial: int, success: bool = False,
                        output_dir: str = None,
                        experiment: str = "gap", platform: str = "ardupilot"):
        """Export the newest ArduPilot BIN log under GAP's unified filename scheme."""
        from evaluation.common.metrics import export_raw_log, make_output_path, raw_log_mode

        src_log_dir = os.path.join(ARDUPILOT_ROOT, "ArduCopter", "logs")
        if not os.path.exists(src_log_dir):
            logger.warning(f"[Ardupilot-{self.frame_type}] Log directory not found: {src_log_dir}")
            return None
        bin_files = glob.glob(os.path.join(src_log_dir, "*.BIN"))
        if not bin_files:
            logger.warning(f"[Ardupilot-{self.frame_type}] No .BIN files in {src_log_dir}")
            return None
        latest_log = max(bin_files, key=os.path.getmtime)

        if output_dir is None:
            output_dir = os.environ.get("GAP_ARDUPILOT_LOG_DIR") or \
                os.path.join(LOG_DIR, "flight-logs", "ardupilot", "raw")
        os.makedirs(output_dir, exist_ok=True)

        dest_path = make_output_path(
            output_dir,
            experiment=experiment,
            platform=platform,
            frame=self.frame_type,
            worker=position_idx + 1,
            episode=trial,
            outcome=("success" if success else "fail"),
            ext="BIN",
        )
        try:
            action = export_raw_log(latest_log, dest_path)
            if action == "deleted":
                logger.info(f"[Ardupilot-{self.frame_type}] Flight log deleted (GAP_RAW_LOG_MODE=off)")
                return None
            logger.info(
                f"[Ardupilot-{self.frame_type}] Flight log {action}: {os.path.basename(dest_path)} "
                f"(GAP_RAW_LOG_MODE={raw_log_mode()})"
            )
            return dest_path
        except Exception as e:
            logger.error(f"[Ardupilot-{self.frame_type}] Failed to copy log: {e}")
            return None

    def clear_failsafe_events(self):
        """Clear recorded failsafe events."""
        self.failsafe_events = []
        logger.debug(f"[Ardupilot-{self.frame_type}] Failsafe events cleared")

    def check_failsafe(self) -> list:
        """Collect newly arrived failsafe STATUSTEXT events."""
        new_events = []

        while True:
            msg = self.master.recv_match(type='STATUSTEXT', blocking=False)
            if msg is None:
                break

            event = self._record_failsafe_text(msg.text if hasattr(msg, 'text') else str(msg))
            if event is not None:
                new_events.append(event)

        return new_events

    def get_failsafe_events(self) -> list:
        """Return all recorded failsafe events."""
        return self.failsafe_events.copy()

    def has_failsafe_occurred(self) -> bool:
        """Return whether any failsafe has been recorded."""
        return len(self.failsafe_events) > 0
