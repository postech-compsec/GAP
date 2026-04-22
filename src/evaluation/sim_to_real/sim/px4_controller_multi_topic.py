"""PX4 sim-to-real SITL controller using GPS_RAW_INT + ATTITUDE_QUATERNION + SCALED_IMU."""

import os, time, subprocess, math, signal, shutil, glob
os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil
import numpy as np
from config import STANDARD_GRAVITY, PX4_ROOT, PX4_RUN_SCRIPT, logger, PX4_PARAM_SPEED


class PX4MultiTopicController:

    def __init__(self, instance_id: int, speed_factor="1", headless="0"):
        self.instance_id = instance_id
        self.url = f'udpin:0.0.0.0:{17000 + self.instance_id}'
        self.master = mavutil.mavlink_connection(self.url)

        self.speed_factor = speed_factor
        self.mav_recv_timeout = 48 / int(speed_factor)
        self.mav_short_timeout = 5 / int(speed_factor)

        try:
            self.master.wait_heartbeat(timeout=self.mav_short_timeout)
            logger.info(f"[PX4MultiTopic] Heartbeat received: sys={self.master.target_system}, comp={self.master.target_component}")
        except Exception as exc:
            logger.warning(f"[PX4MultiTopic] Heartbeat wait failed: {exc}")

        self.proc = None
        self.headless = headless
        self.home_coordinates = {}

        # Mirror the rates seen by the EKF, not the uncorrected IMU stream.
        self.current_gyro_bias = np.zeros(3, dtype=np.float32)
        self.prev_gps_alt = None
        self.prev_gps_timestamp = None

        _this = os.path.abspath(__file__)
        _project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this))))
        )
        _default_ulog = os.path.join(_project_root, "results", "fresh", "flight-logs", "px4", "raw")
        self.ulog_base_dir = os.environ.get("GAP_PX4_LOG_DIR") or os.environ.get("GAP_ULOG_DIR", _default_ulog)
        self._episode_ulog_snapshot = set()
        self._episode_ulog_started_at = None

    def _instance_log_base(self):
        return os.path.join(
            PX4_ROOT, "build", "px4_sitl_default",
            f"instance_{self.instance_id}", "log"
        )

    def _list_instance_ulogs(self):
        px4_instance_log_base = self._instance_log_base()
        if not os.path.exists(px4_instance_log_base):
            return []
        return sorted(glob.glob(os.path.join(px4_instance_log_base, "*", "*.ulg")))

    def start_px4(self, latitude: float = None, longitude: float = None) -> subprocess.Popen:

        if not os.path.exists(PX4_RUN_SCRIPT):
            raise FileNotFoundError(f"PX4 run script not found at {PX4_RUN_SCRIPT}")

        try:
            subprocess.run(["pkill", "-9", "-f", f":{17000 + self.instance_id}"],
                          capture_output=True, timeout=3)
            time.sleep(0.5)
            logger.debug(f"[PX4-{self.instance_id}] Cleaned up stale processes on port {17000 + self.instance_id}")
        except Exception as e:
            logger.debug(f"[PX4-{self.instance_id}] Port cleanup error (non-fatal): {e}")

        env = dict(os.environ, PX4_SIM_SPEED_FACTOR=self.speed_factor, HEADLESS=self.headless)
        if latitude is not None and longitude is not None:
            env['PX4_HOME_LAT'] = str(latitude)
            env['PX4_HOME_LON'] = str(longitude)
        logger.debug(f"[PX4-{self.instance_id}] Modifying start location: LAT={latitude}, LON={longitude}")

        cmd = ["/usr/bin/env", "bash", PX4_RUN_SCRIPT, str(self.instance_id)]

        self._episode_ulog_snapshot = set(self._list_instance_ulogs())
        self._episode_ulog_started_at = time.time()

        self.proc = subprocess.Popen(cmd,
                                     env=env,
                                     cwd=PX4_ROOT,
                                     preexec_fn=os.setsid,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        
        self.home_coordinates = {}

        logger.info(f"[PX4-{self.instance_id}] Launched PX4 process PID={self.proc.pid}")

        return self.proc

    def stop(self):
        if self.proc and self.proc.poll() is None:
            pid = self.proc.pid
            pgid = os.getpgid(pid)
            logger.info(f"[{self.__class__.__name__}-{self.instance_id}] Stopping process group PGID={pgid}")

            try:
                os.killpg(pgid, signal.SIGINT)
                self.proc.wait(timeout=10)
                logger.info(f"[{self.__class__.__name__}-{self.instance_id}] Process group PGID={pgid} stopped gracefully.")
                self.proc = None
                return
            except (subprocess.TimeoutExpired, ProcessLookupError):
                pass

            logger.warning(f"[{self.__class__.__name__}-{self.instance_id}] Graceful shutdown failed for PGID={pgid}. Killing.")
            try:
                os.killpg(pgid, signal.SIGKILL)
                subprocess.run(["pkill", "-9", "-f", f":{17000 + self.instance_id}"], capture_output=True, timeout=3)
            except ProcessLookupError:
                pass

            t0 = time.time()
            while time.time() - t0 < 10:
                if self.proc.poll() is not None:
                    logger.info(f"[{self.__class__.__name__}-{self.instance_id}] Process group PGID={pgid} confirmed terminated.")
                    self.proc = None
                    return
                time.sleep(0.1)

            logger.error(f"[{self.__class__.__name__}-{self.instance_id}] FAILED to confirm termination of PGID={pgid}.")

        self.proc = None

    def wait_px4_ready(self, timeout):

        logger.debug(f"[PX4-{self.instance_id}] Waiting for PX4 heartbeat on {self.url}...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=self.mav_recv_timeout)
            if msg is not None:
                logger.info(f"[PX4-{self.instance_id}] Heartbeat received.")
                return True
        logger.error(f"[PX4-{self.instance_id}] Timeout waiting for PX4 heartbeat.")
        return False

    def wait_health_ok(self, timeout):

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
                return True, self.home_coordinates

        return False, self.home_coordinates

    def arm_and_takeoff(self, target_alt_m, hover_time=10, max_retries=3, retry_delay_s=1):
        logger.info(f"[PX4-{self.instance_id}] arm_and_takeoff called: target_alt={target_alt_m:.2f}m, hover_time={hover_time:.1f}s, max_retries={max_retries}")
        logger.debug(f"[PX4-{self.instance_id}] MAVLink timeouts: recv={self.mav_recv_timeout:.2f}s, short={self.mav_short_timeout:.2f}s")

        for attempt in range(1, max_retries+1):
            logger.debug(f"[PX4-{self.instance_id}] Arm/takeoff attempt {attempt}/{max_retries}: waiting {retry_delay_s}s before starting")
            time.sleep(retry_delay_s)

            logger.debug(f"[PX4-{self.instance_id}] Checking EXTENDED_SYS_STATE (timeout={self.mav_recv_timeout:.2f}s)")
            msg = self.master.recv_match(type="EXTENDED_SYS_STATE", blocking=True, timeout=self.mav_recv_timeout)
            if msg:
                logger.debug(f"[PX4-{self.instance_id}] EXTENDED_SYS_STATE received: landed_state={msg.landed_state}")
                if msg.landed_state == mavutil.mavlink.MAV_LANDED_STATE_TAKEOFF:
                    logger.info(f"[PX4-{self.instance_id}] Vehicle is already flying. Skipping arm and takeoff sequence.")
                    break
            else:
                logger.warning(f"[PX4-{self.instance_id}] No EXTENDED_SYS_STATE message received (timeout)")

            PX4_MAIN_AUTO = 4
            PX4_SUB_TAKEOFF = 2
            cm = PX4_MAIN_AUTO | (PX4_SUB_TAKEOFF << 8)
            logger.debug(f"[PX4-{self.instance_id}] Setting mode to AUTO.TAKEOFF (custom_mode={cm})")
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                cm
            )
            sleep_duration = 2/int(PX4_PARAM_SPEED)
            logger.debug(f"[PX4-{self.instance_id}] Waiting {sleep_duration:.2f}s for mode change")
            time.sleep(sleep_duration)

            # Drop stale ACKs between commands.
            drained = self.drain_all()
            logger.debug(f"[PX4-{self.instance_id}] Drained {drained} messages before ARM command")

            logger.info(f"[PX4-{self.instance_id}] Sending ARM command (Attempt {attempt}/{max_retries})")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )

            logger.debug(f"[PX4-{self.instance_id}] Waiting for ARM ACK (timeout={self.mav_recv_timeout:.2f}s)")
            arm_ack = None
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.mav_recv_timeout:
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1.0)
                if msg and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    arm_ack = msg
                    break
                elif msg:
                    logger.debug(f"[PX4-{self.instance_id}] Ignoring stale ACK: command={msg.command}, result={msg.result}")

            if arm_ack:
                logger.debug(f"[PX4-{self.instance_id}] ARM ACK received: command={arm_ack.command}, result={arm_ack.result}")
            else:
                logger.error(f"[PX4-{self.instance_id}] ARM ACK timeout after {self.mav_recv_timeout:.2f}s")

            if not (arm_ack and arm_ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
                logger.warning(f"[PX4-{self.instance_id}] Arming failed with ACK result: {arm_ack.result if arm_ack else 'Timeout'}. Retrying entire sequence...")
                continue
            logger.info(f"[PX4-{self.instance_id}] ✓ Vehicle armed successfully!")

            sleep_duration = 2/int(PX4_PARAM_SPEED)
            logger.debug(f"[PX4-{self.instance_id}] Waiting {sleep_duration:.2f}s before takeoff command")
            time.sleep(sleep_duration)

            drained = self.drain_all()
            logger.debug(f"[PX4-{self.instance_id}] Drained {drained} messages before TAKEOFF command")

            logger.info(f"[PX4-{self.instance_id}] Sending TAKEOFF command to {target_alt_m:.2f}m")
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0, 0, 0, math.nan,
                math.nan, math.nan,
                target_alt_m
            )

            logger.debug(f"[PX4-{self.instance_id}] Waiting for TAKEOFF ACK (timeout={self.mav_recv_timeout:.2f}s)")
            takeoff_ack = None
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.mav_recv_timeout:
                msg = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=1.0)
                if msg and msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                    takeoff_ack = msg
                    break
                elif msg:
                    logger.debug(f"[PX4-{self.instance_id}] Ignoring stale ACK: command={msg.command}, result={msg.result}")

            if takeoff_ack:
                logger.debug(f"[PX4-{self.instance_id}] TAKEOFF ACK received: command={takeoff_ack.command}, result={takeoff_ack.result}")
            else:
                logger.error(f"[PX4-{self.instance_id}] TAKEOFF ACK timeout after {self.mav_recv_timeout:.2f}s")

            if not (takeoff_ack and takeoff_ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED):
                logger.warning(f"[PX4-{self.instance_id}] Takeoff failed with ACK result: {takeoff_ack.result if takeoff_ack else 'Timeout'}. Retrying entire sequence...")
                continue
            logger.info(f"[PX4-{self.instance_id}] ✓ Arm and Takeoff commands accepted!")

            break
        else:
            logger.error(f"[PX4-{self.instance_id}] *** Failed to complete arm and takeoff sequence after {max_retries} retries ***")
            raise RuntimeError("Failed to complete arm and takeoff sequence after multiple retries.")

        logger.info(f"[PX4-{self.instance_id}] Waiting for vehicle to reach target altitude {target_alt_m:.2f}m (timeout={hover_time:.1f}s)")
        t0 = time.monotonic()
        last_log_time = t0
        check_count = 0

        while time.monotonic() - t0 < hover_time:
            msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=self.mav_recv_timeout)
            check_count += 1

            if msg:
                current_altitude = msg.alt / 1000.0
                elapsed = time.monotonic() - t0

                if time.monotonic() - last_log_time >= 2.0:
                    logger.debug(f"[PX4-{self.instance_id}] Altitude check #{check_count}: current={current_altitude:.2f}m, target={target_alt_m:.2f}m, elapsed={elapsed:.1f}s")
                    last_log_time = time.monotonic()

                if current_altitude >= target_alt_m - 1:
                    logger.info(f"[PX4-{self.instance_id}] ✓ Vehicle reached target altitude: {current_altitude:.2f}m (took {elapsed:.1f}s, {check_count} checks)")
                    return True
            else:
                logger.warning(f"[PX4-{self.instance_id}] GLOBAL_POSITION_INT timeout during altitude check #{check_count}")

        final_elapsed = time.monotonic() - t0
        logger.error(f"[PX4-{self.instance_id}] ✗ Vehicle failed to reach target altitude after {final_elapsed:.1f}s ({check_count} checks)")
        return False

    def get_hil(self, timeout):
        """Assemble a HIL-equivalent 16-vector from GPS_RAW_INT, ATTITUDE_QUATERNION, and SCALED_IMU."""
        required_types = ['GPS_RAW_INT', 'ATTITUDE_QUATERNION', 'SCALED_IMU']
        latest_msgs = {msg_type: None for msg_type in required_types}

        start = time.monotonic()
        deadline = start + timeout if timeout is not None else None

        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
            else:
                remaining = None

            msg = self.master.recv_match(
                type=required_types,
                blocking=True,
                timeout=None if remaining is None else max(0.0, remaining),
            )
            if msg is None:
                break

            msg_type = msg.get_type()
            if msg_type in latest_msgs:
                latest_msgs[msg_type] = msg

            if all(latest_msgs[mt] is not None for mt in required_types):
                break

        if not all(latest_msgs[mt] is not None for mt in required_types):
            return None

        gps_msg = latest_msgs['GPS_RAW_INT']
        att_msg = latest_msgs['ATTITUDE_QUATERNION']
        imu_msg = latest_msgs['SCALED_IMU']

        gps_ground_speed_m_s = gps_msg.vel / 100.0
        gps_cog_rad = math.radians(gps_msg.cog / 100.0)
        vx_gps = gps_ground_speed_m_s * math.cos(gps_cog_rad)
        vy_gps = gps_ground_speed_m_s * math.sin(gps_cog_rad)

        current_alt = gps_msg.alt / 1000.0
        current_timestamp = gps_msg.time_usec / 1e6
        if self.prev_gps_alt is not None and self.prev_gps_timestamp is not None:
            dt = current_timestamp - self.prev_gps_timestamp
            vz_ned = (current_alt - self.prev_gps_alt) / dt if dt > 0.0 else 0.0
        else:
            vz_ned = 0.0
        self.prev_gps_alt = current_alt
        self.prev_gps_timestamp = current_timestamp

        xacc = imu_msg.xacc * STANDARD_GRAVITY / 1000.0
        yacc = imu_msg.yacc * STANDARD_GRAVITY / 1000.0
        zacc = imu_msg.zacc * STANDARD_GRAVITY / 1000.0
        xgyro = (imu_msg.xgyro / 1000.0) - self.current_gyro_bias[0]
        ygyro = (imu_msg.ygyro / 1000.0) - self.current_gyro_bias[1]
        zgyro = (imu_msg.zgyro / 1000.0) - self.current_gyro_bias[2]

        arr = np.array([
            gps_msg.lat * 1e-7, gps_msg.lon * 1e-7, current_alt,
            vx_gps, vy_gps, vz_ned,
            xacc, yacc, zacc,
            att_msg.q1, att_msg.q2, att_msg.q3, att_msg.q4,
            xgyro, ygyro, zgyro,
        ], np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def get_gyro_bias(self, timeout=1.0):

        msg = self.master.recv_match(type='GET_GYRO_BIAS', blocking=True, timeout=timeout)
        if not msg:
            return None
            
        arr = np.array([
            msg.gyro_bias_x, msg.gyro_bias_y, msg.gyro_bias_z
        ], np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        return arr
    
    def set_gyro_bias(self, bias):
        self.master.mav.set_gyro_bias_send(bias[0], bias[1], bias[2])
        self.current_gyro_bias = np.array(bias, dtype=np.float32)

    def disarm(self):
        """Disarm the vehicle."""
        logger.info(f"[PX4-{self.instance_id}] Disarming vehicle")
        self.master.mav.command_long_send(
            self.master.target_system, 
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )

        disarm_ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=2)
        if disarm_ack and disarm_ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if disarm_ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                logger.info(f"[PX4-{self.instance_id}] Vehicle disarmed successfully")
                return True
            else:
                logger.warning(f"[PX4-{self.instance_id}] Disarm command rejected: {disarm_ack.result}")
        else:
            logger.warning(f"[PX4-{self.instance_id}] No disarm acknowledgment received")
        
        return False

    def drain_all(self):
        drained = 0
        while True:
            m = self.master.recv_match(blocking=False)
            if m is None:
                break
            drained += 1

        if drained > 100:
            logger.debug(f"[PX4-{self.instance_id}] Drained {drained} messages from the MAVLink connection.")

        return drained

    def organize_ulog_files(self, success: bool, episode_num: int,
                            variant=None, experiment: str = "gap",
                            platform: str = "px4-jmavsim"):
        """Export the newest ULog under GAP's unified filename scheme."""
        from evaluation.common.metrics import export_raw_log, make_output_path, raw_log_mode

        try:
            px4_instance_log_base = self._instance_log_base()
            if not os.path.exists(px4_instance_log_base):
                logger.warning(f"[PX4-{self.instance_id}] Instance log dir not found: {px4_instance_log_base}")
                return

            ulog_files = self._list_instance_ulogs()
            if not ulog_files:
                logger.warning(f"[PX4-{self.instance_id}] No .ulg files in {px4_instance_log_base}")
                return

            snapshot = getattr(self, "_episode_ulog_snapshot", set()) or set()
            candidates = [p for p in ulog_files if p not in snapshot]
            if not candidates and self._episode_ulog_started_at is not None:
                candidates = [
                    p for p in ulog_files
                    if os.path.getmtime(p) >= self._episode_ulog_started_at - 1.0
                ]
            if not candidates:
                logger.warning(
                    f"[PX4-{self.instance_id}] No new .ulg files matched the current episode; "
                    f"falling back to newest file."
                )
                candidates = ulog_files
            if len(candidates) > 1:
                logger.warning(
                    f"[PX4-{self.instance_id}] Multiple episode ULog candidates found; "
                    f"choosing newest of {len(candidates)} files."
                )

            latest_ulog = max(candidates, key=os.path.getmtime)
            os.makedirs(self.ulog_base_dir, exist_ok=True)

            dest_path = make_output_path(
                self.ulog_base_dir,
                experiment=experiment,
                platform=platform,
                variant=variant,
                worker=self.instance_id,
                episode=episode_num,
                outcome=("success" if success else "fail"),
                ext="ulg",
            )
            action = export_raw_log(latest_ulog, dest_path)
            if action == "deleted":
                logger.info(f"[PX4-{self.instance_id}] Ulog deleted (GAP_RAW_LOG_MODE=off)")
            else:
                logger.info(
                    f"[PX4-{self.instance_id}] Ulog {action}: {os.path.basename(dest_path)} "
                    f"(GAP_RAW_LOG_MODE={raw_log_mode()})"
                )
            self._episode_ulog_snapshot = set(self._list_instance_ulogs())
            self._episode_ulog_started_at = None

        except Exception as e:
            logger.error(f"[PX4-{self.instance_id}] Failed to organize ulog file: {e}", exc_info=True)
