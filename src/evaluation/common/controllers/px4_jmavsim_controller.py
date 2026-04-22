"""PX4 jMAVSim controller used by GAP evaluation."""

import os
import time
import subprocess
import math
import signal
import glob
import numpy as np

os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil

from ..test_config import STANDARD_GRAVITY, PX4_ROOT, LOG_DIR, logger, SIMULATION_SPEEDUP


class PX4JMAVSimController:
    """PX4 jMAVSim SITL lifecycle and MAVLink helper."""

    def __init__(self, instance_id: int = 0, speed_factor: int = None, headless: bool = True):
        """Create one controller for one PX4 jMAVSim instance."""
        self.instance_id = instance_id
        self.speed_factor = speed_factor if speed_factor is not None else SIMULATION_SPEEDUP
        self.headless = headless

        self.url = f'udpin:0.0.0.0:{17000 + self.instance_id}'
        self.master = mavutil.mavlink_connection(self.url)

        self.mav_recv_timeout = 48 / self.speed_factor
        self.mav_short_timeout = 5 / self.speed_factor

        try:
            self.master.wait_heartbeat(timeout=self.mav_short_timeout)
            logger.info(f"[PX4JMAVSim-{self.instance_id}] Heartbeat received")
        except Exception as exc:
            logger.warning(f"[PX4JMAVSim-{self.instance_id}] Heartbeat wait failed: {exc}")

        self.proc = None
        self.home_coordinates = {}
        self.ulog_base_dir = (
            os.environ.get("GAP_PX4_LOG_DIR")
            or os.path.join(LOG_DIR, "flight-logs", "px4", "raw")
        )
        self._episode_ulog_snapshot = set()
        self._episode_ulog_started_at = None

    def _instance_log_base(self):
        if self.instance_id == 0:
            return os.path.join(PX4_ROOT, "build", "px4_sitl_default", "rootfs", "log")
        return os.path.join(
            PX4_ROOT, "build", "px4_sitl_default",
            f"instance_{self.instance_id}", "log",
        )

    def _list_instance_ulogs(self):
        px4_instance_log_base = self._instance_log_base()
        if not os.path.exists(px4_instance_log_base):
            return []
        return sorted(glob.glob(os.path.join(px4_instance_log_base, "*", "*.ulg")))

    def _collect_episode_ulog_candidates(self, wait_timeout: float = 5.0):
        """Wait briefly for PX4 to flush the current episode's ULog."""
        snapshot = getattr(self, "_episode_ulog_snapshot", set()) or set()
        deadline = time.monotonic() + wait_timeout

        while True:
            ulog_files = self._list_instance_ulogs()
            candidates = [p for p in ulog_files if p not in snapshot]

            if not candidates and self._episode_ulog_started_at is not None:
                candidates = [
                    p for p in ulog_files
                    if os.path.getmtime(p) >= self._episode_ulog_started_at - 1.0
                ]

            if candidates or time.monotonic() >= deadline:
                return ulog_files, candidates

            time.sleep(0.2)

    def start(self, latitude: float = None, longitude: float = None, altitude: float = 485.0):
        """Start one PX4 jMAVSim SITL instance."""
        try:
            subprocess.run(["pkill", "-9", "-f", "jmavsim"],
                          capture_output=True, timeout=3)
            time.sleep(0.5)
            logger.debug(f"[PX4JMAVSim-{self.instance_id}] Cleaned up stale jMAVSim processes")
        except Exception as e:
            logger.debug(f"[PX4JMAVSim-{self.instance_id}] Process cleanup error (non-fatal): {e}")

        env = dict(os.environ)
        env['PX4_SIM_SPEED_FACTOR'] = str(self.speed_factor)

        if latitude is not None and longitude is not None:
            env['PX4_HOME_LAT'] = str(latitude)
            env['PX4_HOME_LON'] = str(longitude)
            env['PX4_HOME_ALT'] = str(altitude)
            logger.info(f"[PX4JMAVSim-{self.instance_id}] Setting home position: "
                       f"LAT={latitude}, LON={longitude}, ALT={altitude}m")

        cmd = [
            "make",
            "px4_sitl_default",
            "jmavsim"
        ]

        if self.headless:
            env['HEADLESS'] = '1'

        logger.info(f"[PX4JMAVSim-{self.instance_id}] Starting PX4-jMAVSim with speed factor {self.speed_factor}")
        self._episode_ulog_snapshot = set(self._list_instance_ulogs())
        self._episode_ulog_started_at = time.time()

        self.proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=PX4_ROOT,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        self.home_coordinates = {}
        logger.info(f"[PX4JMAVSim-{self.instance_id}] Launched PX4-jMAVSim process PID={self.proc.pid}")

        time.sleep(5 / self.speed_factor)

        try:
            self.master.close()
        except:
            pass

        self.master = mavutil.mavlink_connection(self.url)
        logger.debug(f"[PX4JMAVSim-{self.instance_id}] Re-established MAVLink connection")

        return self.proc

    def stop(self):
        """Stop the PX4 jMAVSim instance."""
        if self.proc and self.proc.poll() is None:
            pid = self.proc.pid
            pgid = os.getpgid(pid)
            logger.info(f"[PX4JMAVSim-{self.instance_id}] Stopping process group PGID={pgid}")

            try:
                os.killpg(pgid, signal.SIGINT)
                self.proc.wait(timeout=10)
                logger.info(f"[PX4JMAVSim-{self.instance_id}] Process stopped gracefully")
                self.proc = None
                return
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass

            logger.warning(f"[PX4JMAVSim-{self.instance_id}] Graceful shutdown failed. Killing.")
            try:
                os.killpg(pgid, signal.SIGKILL)
                subprocess.run(["pkill", "-9", "-f", "jmavsim"],
                             capture_output=True, timeout=3)
            except ProcessLookupError:
                pass

            t0 = time.time()
            while time.time() - t0 < 10:
                if self.proc.poll() is not None:
                    logger.info(f"[PX4JMAVSim-{self.instance_id}] Process confirmed terminated")
                    self.proc = None
                    return
                time.sleep(0.1)

            logger.error(f"[PX4JMAVSim-{self.instance_id}] Failed to confirm termination")

        self.proc = None

    def wait_ready(self, timeout: float = 60):
        """Wait for the first heartbeat."""
        logger.debug(f"[PX4JMAVSim-{self.instance_id}] Waiting for heartbeat...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=self.mav_recv_timeout)
            if msg is not None:
                logger.info(f"[PX4JMAVSim-{self.instance_id}] Heartbeat received")
                return True
        logger.error(f"[PX4JMAVSim-{self.instance_id}] Timeout waiting for heartbeat")
        return False

    def wait_health_ok(self, timeout: float = 60):
        """Wait until position, home, and GPS are available."""
        got_global = got_home = got_gps = False

        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            msg = self.master.recv_match(
                type=['GLOBAL_POSITION_INT', 'HOME_POSITION', 'GPS_RAW_INT'],
                blocking=True, timeout=self.mav_recv_timeout
            )
            if not msg:
                continue
            t = msg.get_type()

            if t == 'GLOBAL_POSITION_INT':
                lat = msg.lat / 1e7
                lon = msg.lon / 1e7
                if lat != 0.0 or lon != 0.0:
                    got_global = True
            elif t == 'HOME_POSITION':
                self.home_coordinates = {
                    'lat': msg.latitude / 1e7,
                    'lon': msg.longitude / 1e7,
                    'alt': msg.altitude / 1000.0
                }
                got_home = True
            elif t == 'GPS_RAW_INT':
                if msg.fix_type >= 3:
                    got_gps = True

            if got_global and got_home and got_gps:
                logger.info(f"[PX4JMAVSim-{self.instance_id}] Health OK. Home position: "
                           f"LAT={self.home_coordinates['lat']:.6f}, LON={self.home_coordinates['lon']:.6f}, "
                           f"ALT={self.home_coordinates['alt']:.2f}m")
                return True, self.home_coordinates

        return False, self.home_coordinates

    def arm_and_takeoff(self, target_alt_m: float, hover_time: float = 60, max_retries: int = 3):
        """Arm and climb to the requested altitude."""
        logger.info(f"[PX4JMAVSim-{self.instance_id}] Arm and takeoff to {target_alt_m:.2f}m")

        for attempt in range(1, max_retries + 1):
            time.sleep(1 / self.speed_factor)

            msg = self.master.recv_match(type="EXTENDED_SYS_STATE", blocking=True, timeout=self.mav_recv_timeout)
            if msg and msg.landed_state == mavutil.mavlink.MAV_LANDED_STATE_TAKEOFF:
                logger.info(f"[PX4JMAVSim-{self.instance_id}] Already flying")
                break

            PX4_MAIN_AUTO = 4
            PX4_SUB_TAKEOFF = 2
            cm = PX4_MAIN_AUTO | (PX4_SUB_TAKEOFF << 8)
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                cm
            )
            time.sleep(2 / self.speed_factor)

            self.drain_all()

            logger.info(f"[PX4JMAVSim-{self.instance_id}] Sending ARM command (attempt {attempt}/{max_retries})")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )

            arm_ack = None
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.mav_recv_timeout:
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1.0)
                if msg and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    arm_ack = msg
                    break

            if not (arm_ack and arm_ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
                logger.warning(f"[PX4JMAVSim-{self.instance_id}] Arming failed, retrying...")
                continue

            logger.info(f"[PX4JMAVSim-{self.instance_id}] Vehicle armed")
            time.sleep(2 / self.speed_factor)

            self.drain_all()

            logger.info(f"[PX4JMAVSim-{self.instance_id}] Sending TAKEOFF command")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, math.nan, math.nan, math.nan, target_alt_m
            )

            takeoff_ack = None
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.mav_recv_timeout:
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1.0)
                if msg and msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                    takeoff_ack = msg
                    break

            if not (takeoff_ack and takeoff_ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
                logger.warning(f"[PX4JMAVSim-{self.instance_id}] Takeoff failed, retrying...")
                continue

            logger.info(f"[PX4JMAVSim-{self.instance_id}] Arm and takeoff commands accepted")
            break
        else:
            logger.error(f"[PX4JMAVSim-{self.instance_id}] Failed after {max_retries} retries")
            raise RuntimeError("Failed to arm and takeoff")

        logger.info(f"[PX4JMAVSim-{self.instance_id}] Waiting for altitude {target_alt_m:.2f}m")
        t0 = time.monotonic()
        while time.monotonic() - t0 < hover_time:
            msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=self.mav_recv_timeout)
            if msg:
                current_altitude = msg.alt / 1000.0
                if current_altitude >= target_alt_m - 1:
                    logger.info(f"[PX4JMAVSim-{self.instance_id}] Target altitude reached: {current_altitude:.2f}m")
                    return True

        logger.error(f"[PX4JMAVSim-{self.instance_id}] Failed to reach target altitude")
        return False

    def get_hil(self, timeout: float = 1.0):
        """Return one HIL_STATE_QUATERNION sample as a float32 array."""
        msg = self.master.recv_match(type='HIL_STATE_QUATERNION', blocking=True, timeout=timeout)
        if not msg:
            return None

        arr = np.array([
            msg.lat * 1e-7, msg.lon * 1e-7, msg.alt / 1000.0,
            msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0,
            (msg.xacc / 1000.0) * STANDARD_GRAVITY,
            (msg.yacc / 1000.0) * STANDARD_GRAVITY,
            (msg.zacc / 1000.0) * STANDARD_GRAVITY,
            *msg.attitude_quaternion,
            msg.rollspeed, msg.pitchspeed, msg.yawspeed,
        ], np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def set_gyro_bias(self, bias: np.ndarray):
        """Send the external gyro bias."""
        self.master.mav.set_gyro_bias_send(bias[0], bias[1], bias[2])

    def get_gyro_bias(self, timeout: float = 1.0):
        """Poll the current external gyro bias."""
        msg = self.master.recv_match(type='GET_GYRO_BIAS', blocking=True, timeout=timeout)
        if not msg:
            return None

        arr = np.array([msg.gyro_bias_x, msg.gyro_bias_y, msg.gyro_bias_z], np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def drain_all(self):
        """Drain buffered MAVLink messages."""
        drained = 0
        while True:
            m = self.master.recv_match(blocking=False)
            if m is None:
                break
            drained += 1

        if drained > 100:
            logger.debug(f"[PX4JMAVSim-{self.instance_id}] Drained {drained} messages")

        return drained

    def copy_flight_log(self, position_idx: int, trial: int, success: bool = False,
                        output_dir: str = None,
                        experiment: str = "gap", platform: str = "px4-jmavsim"):
        """Export the newest PX4 ULog under GAP's unified filename scheme."""
        from evaluation.common.metrics import export_raw_log, make_output_path, raw_log_mode

        px4_instance_log_base = self._instance_log_base()
        if not os.path.exists(px4_instance_log_base):
            logger.warning(f"[PX4JMAVSim-{self.instance_id}] Instance log dir not found: {px4_instance_log_base}")
            return None

        ulog_files, candidates = self._collect_episode_ulog_candidates()
        if not ulog_files:
            logger.warning(f"[PX4JMAVSim-{self.instance_id}] No .ulg files in {px4_instance_log_base}")
            return None

        if not candidates:
            logger.warning(
                f"[PX4JMAVSim-{self.instance_id}] No new .ulg files matched the current episode; "
                f"falling back to newest file."
            )
            candidates = ulog_files

        if len(candidates) > 1:
            logger.warning(
                f"[PX4JMAVSim-{self.instance_id}] Multiple episode ULog candidates found; "
                f"choosing newest of {len(candidates)} files."
            )

        latest_ulog = max(candidates, key=os.path.getmtime)
        if output_dir is None:
            output_dir = self.ulog_base_dir
        os.makedirs(output_dir, exist_ok=True)

        dest_path = make_output_path(
            output_dir,
            experiment=experiment,
            platform=platform,
            worker=position_idx + 1,
            episode=trial,
            outcome=("success" if success else "fail"),
            ext="ulg",
        )
        try:
            action = export_raw_log(latest_ulog, dest_path)
            if action == "deleted":
                logger.info(f"[PX4JMAVSim-{self.instance_id}] Ulog deleted (GAP_RAW_LOG_MODE=off)")
                return None
            logger.info(
                f"[PX4JMAVSim-{self.instance_id}] Ulog {action}: {os.path.basename(dest_path)} "
                f"(GAP_RAW_LOG_MODE={raw_log_mode()})"
            )
            self._episode_ulog_snapshot = set(self._list_instance_ulogs())
            self._episode_ulog_started_at = None
            return dest_path
        except Exception as exc:
            logger.error(f"[PX4JMAVSim-{self.instance_id}] Failed to export ulog: {exc}")
            return None
