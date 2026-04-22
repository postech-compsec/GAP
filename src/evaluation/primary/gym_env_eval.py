import json
import math
import os
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.spatial.transform import Rotation

from config import (
    ACT_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    DISTANCE_K,
    EP_TIMEOUT,
    HIL_STATE_DIM,
    MAV_TIMEOUT,
    MAX_BIAS,
    METERS_PER_DEGREE_LAT,
    PRIVILEGED_INFO_DIM,
    PX4_PARAM_HEADLESS,
    PX4_PARAM_SPEED,
    RANDOM_START_RADIUS_M,
    SAFE_ALT_MAX,
    SAFE_ALT_MIN,
    SAFE_TARGET_DIST,
    STEP_BASE,
    SUCCESS_RADIUS_BONUS,
    SUCCESS_RADIUS_M,
    TAKEOFF_ALT,
    TARGET_LAT,
    TARGET_LON,
    TIME_PENALTY,
    TRUNCATED_PENALTY,
    VELOCITY_K,
    ZERO_ACTION,
    logger,
)
from jmavsim_controller import JmavsimController
from px4_controller import PX4Controller

from evaluation.common.metrics import MetricsCollector, TestResult, NumpyEncoder

# RQ2 noise toggles. Off by default for clean RQ1-style runs.
NOISE_TRACKING = os.environ.get("NOISE_TRACKING", "0") == "1"
NOISE_DELAY_LOSS = os.environ.get("NOISE_DELAY_LOSS", "0") == "1"
PRIMARY_VARIANT = os.environ.get("GAP_PRIMARY_VARIANT", "").strip().lower()


def _noise_variant(tracking: bool, delay_loss: bool):
    """Map (tracking, delay_loss) → RQ2 condition tag used in filenames."""
    if PRIMARY_VARIANT:
        return PRIMARY_VARIANT
    if tracking and delay_loss:
        return "both"
    if tracking:
        return "tracking"
    if delay_loss:
        return "delay-loss"
    return None

class PX4RLEnvEval(gym.Env):
    """PX4-jMAVSim eval env for RQ1/RQ2 with all four success criteria."""

    def _flush_pending_ulog(self):
        if hasattr(self, "_pending_ulog_outcome"):
            success, ep = self._pending_ulog_outcome
            self.px4.organize_ulog_files(
                success=success,
                episode_num=ep,
                variant=_noise_variant(NOISE_TRACKING, NOISE_DELAY_LOSS),
            )
            del self._pending_ulog_outcome

    def __init__(self, env_config):

        super().__init__()

        self.time_penalty = env_config.get('TIME_PENALTY', TIME_PENALTY)
        self.distance_k = env_config.get('DISTANCE_K', DISTANCE_K)
        self.velocity_k = env_config.get('VELOCITY_K', VELOCITY_K)

        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(ACTOR_OBS_DIM,), dtype=np.float32
            ),
            "critic": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(CRITIC_OBS_DIM,), dtype=np.float32
            )
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(ACT_DIM,),
            dtype=np.float32
        )

        self.sensor_noise_std = np.array([
            2.650e-6,  # lat (deg)
            3.922e-6,  # lon (deg)
            0.295,     # alt (m)
            0.417,     # vx (m/s)
            0.417,     # vy (m/s)
            0.417,     # vz (m/s)
            0.0,       # ax
            0.0,       # ay
            0.0,       # az
            0.0,       # qw
            0.0,       # qx
            0.0,       # qy
            0.0,       # qz
            0.026,     # p (rad/s)
            0.026,     # q (rad/s)
            0.026,     # r (rad/s)
        ])
        self.attitude_noise_std_deg = np.array([1.033, 1.033, 1.033])  # roll, pitch, yaw (deg)


        if hasattr(env_config, 'worker_index'):
            self.instance_id = env_config.worker_index
        elif isinstance(env_config, dict) and 'worker_index' in env_config:
            self.instance_id = env_config['worker_index']
        else:
            self.instance_id = 0  # Default fallback

        logger.info(f"[Gym-Eval-{self.instance_id}] Initializing eval env: worker_idx={self.instance_id}")

        _log_dir = os.environ.get("GAP_RESULTS_DIR") or os.environ.get("GAP_LOG_DIR") or "."
        variant = _noise_variant(NOISE_TRACKING, NOISE_DELAY_LOSS)
        self.metrics_collector = MetricsCollector(
            output_dir=_log_dir,
            experiment="gap",
            platform="px4-jmavsim",
            variant=variant,
            worker=self.instance_id,
        )
        self.milestone_interval = 10
        logger.info(
            f"[Gym-Eval-{self.instance_id}] Metrics collector: "
            f"{self.metrics_collector.log_file}"
        )

        sp_list = env_config.get("start_points", None)
        if sp_list:
            wi = self.instance_id
            idx = wi - 1 if 1 <= wi <= len(sp_list) else 0
            pt = sp_list[idx]
            self.fixed_start_lat = pt["lat"]
            self.fixed_start_lon = pt["lon"]
        else:
            self.fixed_start_lat = None
            self.fixed_start_lon = None

        self.px4 = PX4Controller(instance_id=self.instance_id, speed_factor=PX4_PARAM_SPEED, headless=PX4_PARAM_HEADLESS)
        self.jmavsim = JmavsimController(instance_id=self.instance_id, speed_factor=PX4_PARAM_SPEED, headless=PX4_PARAM_HEADLESS)

        self.initial_hil = np.zeros(HIL_STATE_DIM, np.float32)
        self.episode_start_time = 0.0
        self.previous_sub_step_distance_to_target_3d = None

        self.step_times = []
        self.step_times_current = []
        self.last_log_time = None
        self.step_end_time = None
        self.step_count = 0

        self.criteria_success_times = {
            'sphere_20m': None,
            'cylinder_20m': None,
            'sphere_10m': None,
            'cylinder_10m': None
        }
        self.criteria_success_distances = {
            'sphere_20m': None,
            'cylinder_20m': None,
            'sphere_10m': None,
            'cylinder_10m': None
        }

    def reset(self, *, seed=None, options=None, max_retries=5):

        super().reset(seed=seed)
        logger.debug(f"[Gym-Eval-{self.instance_id}] Reset called.")
        self._episode_counter = getattr(self, "_episode_counter", 0)

        self.criteria_success_times = {
            'sphere_20m': None,
            'cylinder_20m': None,
            'sphere_10m': None,
            'cylinder_10m': None
        }
        self.criteria_success_distances = {
            'sphere_20m': None,
            'cylinder_20m': None,
            'sphere_10m': None,
            'cylinder_10m': None
        }

        for attempt in range(max_retries):

            try:
                self.px4.stop()
                self.jmavsim.stop()

                # Export the previous ulog only after PX4 has flushed it.
                if attempt == 0 and hasattr(self, "_pending_ulog_outcome"):
                    self._flush_pending_ulog()

                start_lat, start_lon = None, None
                if options and 'start_lat' in options and 'start_lon' in options:
                    start_lat, start_lon = options['start_lat'], options['start_lon']
                elif self.fixed_start_lat is not None:
                    start_lat, start_lon = self.fixed_start_lat, self.fixed_start_lon
                else:
                    angle_rad = random.uniform(0, 2 * math.pi)
                    offset_north = RANDOM_START_RADIUS_M * math.cos(angle_rad)
                    offset_east = RANDOM_START_RADIUS_M * math.sin(angle_rad)
                    meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(TARGET_LAT))
                    d_lat = offset_north / METERS_PER_DEGREE_LAT
                    d_lon = offset_east / meters_per_degree_lon
                    start_lat = TARGET_LAT + d_lat
                    start_lon = TARGET_LON + d_lon
                    logger.debug(f"[Gym-Eval-{self.instance_id}] Random start position: lat={start_lat:.6f}, lon={start_lon:.6f}, offset_north={offset_north:.2f}m, offset_east={offset_east:.2f}m")

                self.jmavsim.start(latitude=start_lat, longitude=start_lon)
                self.jmavsim.wait_for_ready(timeout=60)
                self.px4.start_px4(latitude=start_lat, longitude=start_lon)

                if not self.px4.wait_px4_ready(timeout=60):
                    raise RuntimeError('PX4 startup failed')
                self.px4.set_gyro_bias(bias=ZERO_ACTION)

                health_ok, _ = self.px4.wait_health_ok(timeout=60)
                if not health_ok:
                    raise RuntimeError('PX4 health check failed')
                self.px4.set_gyro_bias(bias=ZERO_ACTION)

                time.sleep(10)
                self.px4.set_gyro_bias(bias=ZERO_ACTION)

                self.px4.drain_all()
                hil_state = self.px4.get_hil(timeout=(STEP_BASE))
                if hil_state is None:
                    raise RuntimeError('Failed to receive pre-takeoff HIL data')
                self.initial_hil = hil_state.copy()
                logger.debug(f"[Gym-Eval-{self.instance_id}] Initial HIL state received: lat={hil_state[0]:.6f}, lon={hil_state[1]:.6f}, alt={hil_state[2]:.2f}")

                logger.debug(f"[Gym-Eval-{self.instance_id}] Starting takeoff to altitude: {self.initial_hil[2] + TAKEOFF_ALT:.2f}m")
                if not self.px4.arm_and_takeoff(hover_time=60, target_alt_m=self.initial_hil[2] + TAKEOFF_ALT):
                    raise RuntimeError('Takeoff failed')

                self.episode_start_time = time.monotonic()
                logger.debug(f"[Gym-Eval-{self.instance_id}] Episode start time set: {self.episode_start_time}")

                self.px4.drain_all()
                hil_state = self.px4.get_hil(timeout=(STEP_BASE))
                if hil_state is None:
                    raise RuntimeError('Failed to receive post-takeoff HIL data')

                processed_hil_state = self.process_observation(hil_state).copy()
                self.previous_sub_step_distance_to_target_3d = processed_hil_state[13]

                logger.info(f"[Gym-Eval-{self.instance_id}] Environment reset complete.")

                self.step_times.clear()
                self.step_times_current.clear()
                self.last_log_time = time.monotonic()
                self.step_end_time = time.monotonic()
                self.step_count = 0

                return self._build_dict_observation(processed_hil_state, ZERO_ACTION), {}

            except Exception as e:
                logger.debug(f"[Gym-Eval-{self.instance_id}] Reset attempt {attempt+1}/{max_retries} failed: {e}. Retrying...")
                time.sleep(3)

        logger.critical(f"[Gym-Eval-{self.instance_id}] Reset failed after all max_retries. Terminating episode with last known observation.")

        self.step_times.clear()
        self.step_times_current.clear()
        self.last_log_time = time.monotonic()
        self.step_end_time = time.monotonic()
        self.step_count = 0

        dummy_actor_obs = np.zeros(self.observation_space["actor"].shape, dtype=np.float32)
        dummy_critic_obs = np.zeros(self.observation_space["critic"].shape, dtype=np.float32)

        return {"actor": dummy_actor_obs, "critic": dummy_critic_obs}, {'error': 'reset_failed_after_retries'}

    def step(self, action):
        try:
            total_reward = 0.0
            terminated = truncated = False

            self.step_count += 1

            step_start_time = time_previous = time.monotonic()
            bias = np.array([action[0] * MAX_BIAS, action[1] * MAX_BIAS, 0.0])

            if NOISE_DELAY_LOSS:
                bias_injected = random.random() >= 0.125
            else:
                bias_injected = True

            if bias_injected:
                self.px4.set_gyro_bias(bias=bias)
            else:
                logger.debug(f"[Gym-Eval-{self.instance_id}] Bias inject skipped this step")

            if self.step_count % 10 == 0:
                logger.info(f"[Gym-Eval-{self.instance_id}] step {self.step_count}: "
                            f"action=[{action[0]:.3f}, {action[1]:.3f}], "
                            f"bias=[{bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}]")

            self.px4.drain_all()

            # Match the measured delay/loss noise model for RQ2.
            if NOISE_DELAY_LOSS:
                noisy_step_duration = max(
                    STEP_BASE,
                    STEP_BASE + np.random.normal(0.087, 0.076) / int(PX4_PARAM_SPEED),
                )
            else:
                noisy_step_duration = STEP_BASE

            hil_state = None
            processed_hil_state = None
            while time.monotonic() - step_start_time < noisy_step_duration:
                now = time.monotonic()
                delta_time = now - time_previous
                time_previous = now

                sub_hil = self.px4.get_hil(timeout=MAV_TIMEOUT)
                if sub_hil is None:
                    continue

                hil_state = sub_hil
                processed_hil_state = self.process_observation(hil_state).copy()

                reward, terminated, truncated, info = self._calculate_reward(processed_hil_state, delta_time=delta_time)
                self.previous_sub_step_distance_to_target_3d = processed_hil_state[13]

                total_reward += reward

                if truncated:
                    break

            if hil_state is None:
                logger.warning(f"[Gym-Eval-{self.instance_id}] No HIL data received in step; truncating")
                truncated = True
                info = {'reason': 'step_hil_timeout'}
                hil_state = self.initial_hil
                processed_hil_state = self.process_observation(hil_state).copy()

            if NOISE_TRACKING:
                hil_state_for_policy = self.apply_sensor_noise(hil_state)
                processed_hil_state_for_rl = self.process_observation(hil_state_for_policy).copy()
            else:
                hil_state_for_policy = hil_state
                processed_hil_state_for_rl = processed_hil_state

            current_gyro_bias = self.px4.get_gyro_bias(timeout=MAV_TIMEOUT)
            if bias_injected and not terminated and not truncated:
                if current_gyro_bias is None:
                    raise RuntimeError('Failed to receive STEP gyro bias data')
                elif not np.allclose(current_gyro_bias, bias, atol=1e-4):
                    logger.warning(f"[Gym-Eval-{self.instance_id}] Gyro Bias mismatch: expected {bias}, current {current_gyro_bias}")
                    raise RuntimeError(f'Gyro Bias mismatch: expected {bias}, current {current_gyro_bias}')

            dict_obs = self._build_dict_observation(
                processed_hil_state_for_rl, bias
            )

            step_duration = time.monotonic() - self.step_end_time
            self.step_times.append(step_duration)
            self.step_end_time = time.monotonic()
            self.step_times_current.append(self.step_end_time - step_start_time)

            if terminated or truncated:
                if self.step_times and self.step_times_current:
                    avg_step = sum(self.step_times) / len(self.step_times)
                    avg_current = sum(self.step_times_current) / len(self.step_times_current)

                    if len(self.step_times) == len(self.step_times_current):
                        step_time_diffs = [a - b for a, b in zip(self.step_times, self.step_times_current)]
                        avg_diff = sum(step_time_diffs) / len(step_time_diffs)

                        logger.info(f"[Gym-Eval-{self.instance_id}] Episode timing: "
                                   f"avg_step={avg_step:.3f}s, avg_current={avg_current:.3f}s, "
                                   f"avg_diff={avg_diff:.3f}s, steps={len(self.step_times)}")
                    else:
                        logger.info(f"[Gym-Eval-{self.instance_id}] Episode timing: "
                                   f"avg_step={avg_step:.3f}s, avg_current={avg_current:.3f}s, "
                                   f"steps={len(self.step_times)}")

                ep = getattr(self, "_episode_counter", 0) + 1
                self._episode_counter = ep

                time_spent = time.monotonic() - self.episode_start_time

                test_id = len(self.metrics_collector.results) + 1
                result = TestResult(test_id, self.instance_id, ep)
                result.time_spent_s = time_spent
                result.attack_steps = self.step_count

                if 'processed_hil_state' in locals() and 'hil_state' in locals():
                    result.final_distance_m = float(processed_hil_state[13])

                    latitude_position, longitude_position, altitude_amsl = hil_state[0:3]
                    geodesic = Geodesic.WGS84.Inverse(latitude_position, longitude_position, TARGET_LAT, TARGET_LON)
                    horizontal_distance = geodesic['s12']
                    relative_altitude = altitude_amsl - self.initial_hil[2]
                    vertical_distance = relative_altitude - TAKEOFF_ALT

                    result.horizontal_distance_m = float(horizontal_distance)
                    result.vertical_distance_m = float(vertical_distance)
                    result.final_altitude_amsl = float(altitude_amsl)
                    result.final_relative_altitude = float(relative_altitude)
                else:
                    result.final_distance_m = -1.0
                    result.horizontal_distance_m = -1.0
                    result.vertical_distance_m = -1.0

                result.sphere_20m_success = self.criteria_success_times['sphere_20m'] is not None
                result.sphere_20m_time_s = self.criteria_success_times['sphere_20m']
                result.sphere_20m_distance_m = self.criteria_success_distances['sphere_20m']

                result.cylinder_20m_success = self.criteria_success_times['cylinder_20m'] is not None
                result.cylinder_20m_time_s = self.criteria_success_times['cylinder_20m']
                result.cylinder_20m_horizontal_m = self.criteria_success_distances['cylinder_20m']

                result.sphere_10m_success = self.criteria_success_times['sphere_10m'] is not None
                result.sphere_10m_time_s = self.criteria_success_times['sphere_10m']
                result.sphere_10m_distance_m = self.criteria_success_distances['sphere_10m']

                result.cylinder_10m_success = self.criteria_success_times['cylinder_10m'] is not None
                result.cylinder_10m_time_s = self.criteria_success_times['cylinder_10m']
                result.cylinder_10m_horizontal_m = self.criteria_success_distances['cylinder_10m']
                result.terminal_reason = (
                    "success" if result.cylinder_10m_success else info.get("reason")
                )

                self.metrics_collector.add_result(result.to_dict())
                self.metrics_collector.save(print_summary=False)

                self._pending_ulog_outcome = (bool(result.cylinder_10m_success), ep)

                if ep % self.milestone_interval == 0:
                    self.metrics_collector.print_summary()

            return dict_obs, total_reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"[Gym-Eval-{self.instance_id}] step error", exc_info=True)
            dummy_actor_obs = np.zeros(self.observation_space["actor"].shape, dtype=np.float32)
            dummy_critic_obs = np.zeros(self.observation_space["critic"].shape, dtype=np.float32)
            return {"actor": dummy_actor_obs, "critic": dummy_critic_obs}, 0, False, True, {'error': repr(e)}


    def process_observation(self, hil_state):
        latitude_position, longitude_position, altitude_amsl = hil_state[0:3]
        latitude_velocity, longitude_velocity, altitude_velocity = hil_state[3:6]
        quaternion_frd_to_ned = hil_state[9:13]
        roll_speed, pitch_speed, yaw_speed = hil_state[13:16]

        time_since_reset = time.monotonic() - self.episode_start_time
        normalized_time_since_reset = time_since_reset / EP_TIMEOUT

        geodesic = Geodesic.WGS84.Inverse(latitude_position, longitude_position, TARGET_LAT, TARGET_LON)
        direction_to_target_ned = np.deg2rad(geodesic['azi1'])
        distance_to_target = geodesic['s12']
        north_offset_x_to_target = distance_to_target * np.cos(direction_to_target_ned)
        east_offset_y_to_target = distance_to_target * np.sin(direction_to_target_ned)
        relative_altitude = altitude_amsl - self.initial_hil[2]
        z_to_target_downframe = (relative_altitude - TAKEOFF_ALT)
        relative_target_position_ned = np.array([north_offset_x_to_target, east_offset_y_to_target, z_to_target_downframe], np.float32)
        velocity_ned = np.array([latitude_velocity, longitude_velocity, altitude_velocity], np.float32)

        rotation_frd_to_ned = Rotation.from_quat(quaternion_frd_to_ned, scalar_first=True)
        relative_target_position_frd, velocity_frd = rotation_frd_to_ned.apply(np.array([relative_target_position_ned, velocity_ned]),inverse=True)

        roll, pitch, yaw = rotation_frd_to_ned.as_euler('xyz', degrees=False)
        normalized_roll, normalized_pitch, normalized_yaw, heading_error, normalized_direction_to_target_ned = self.normalize_angle(np.array([roll, pitch, yaw, direction_to_target_ned - yaw, direction_to_target_ned]))

        distance_to_target_3d = float(np.sqrt(distance_to_target**2 + z_to_target_downframe**2))
        if distance_to_target_3d > 1e-6:
            target_unit_vec_ned = relative_target_position_ned[:3] / distance_to_target_3d
            closing_velocity_to_target = np.dot(velocity_ned[:3], target_unit_vec_ned)
        else:
            closing_velocity_to_target = 0.0

        distance_to_alt_limit = (relative_altitude - SAFE_ALT_MIN) / (SAFE_ALT_MAX - SAFE_ALT_MIN)
        distance_to_geo_fence = distance_to_target / SAFE_TARGET_DIST

        is_in_sphere = 1 if distance_to_target <= SUCCESS_RADIUS_M else 0

        total_velocity = np.linalg.norm(velocity_frd)

        processed_obs = np.array([
            normalized_time_since_reset,
            *relative_target_position_frd,
            *velocity_frd,
            normalized_roll, normalized_pitch, normalized_yaw,
            roll_speed, pitch_speed, yaw_speed,
            distance_to_target_3d,
            closing_velocity_to_target,
            normalized_direction_to_target_ned,
            heading_error,
            distance_to_alt_limit,
            distance_to_geo_fence,
            is_in_sphere,
            total_velocity,
        ], np.float32)

        return processed_obs

    def _build_dict_observation(self, processed_hil, action):
        actor_obs = np.concatenate([
            processed_hil,
            action
        ]).astype(np.float32)

        processed_priv = np.zeros(PRIVILEGED_INFO_DIM, dtype=np.float32)
        critic_obs = np.concatenate([
            processed_hil,
            processed_priv,
            action
        ]).astype(np.float32)

        return {"actor": actor_obs, "critic": critic_obs}

    def _calculate_reward(self, obs, delta_time=None):
        """Track all four criteria and truncate only on failure/timeout."""
        time_norm = obs[0]
        distance_3d = obs[13]
        closing_vel = obs[14]
        alt_limit = obs[17]
        geo_fence = obs[18]

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.previous_sub_step_distance_to_target_3d is not None:
            progress = self.previous_sub_step_distance_to_target_3d - distance_3d
            reward += progress * self.distance_k

        closing_vel_clipped = np.clip(closing_vel, -5.0, 5.0)
        reward += closing_vel_clipped * self.velocity_k * (delta_time / STEP_BASE)

        reward += self.time_penalty * (delta_time / STEP_BASE)

        current_time = time.monotonic() - self.episode_start_time

        if self.criteria_success_times['sphere_20m'] is None and distance_3d <= 20.0:
            self.criteria_success_times['sphere_20m'] = current_time
            self.criteria_success_distances['sphere_20m'] = distance_3d
            logger.info(f"[Gym-Eval-{self.instance_id}] sphere_20m SUCCESS at {current_time:.2f}s, distance={distance_3d:.2f}m")

        if self.criteria_success_times['sphere_10m'] is None and distance_3d <= 10.0:
            self.criteria_success_times['sphere_10m'] = current_time
            self.criteria_success_distances['sphere_10m'] = distance_3d
            logger.info(f"[Gym-Eval-{self.instance_id}] sphere_10m SUCCESS at {current_time:.2f}s, distance={distance_3d:.2f}m")

        horizontal_distance = float(np.sqrt(obs[1]**2 + obs[2]**2))

        if self.criteria_success_times['cylinder_20m'] is None and horizontal_distance <= 20.0:
            self.criteria_success_times['cylinder_20m'] = current_time
            self.criteria_success_distances['cylinder_20m'] = horizontal_distance
            logger.info(f"[Gym-Eval-{self.instance_id}] cylinder_20m SUCCESS at {current_time:.2f}s, horizontal={horizontal_distance:.2f}m")

        if self.criteria_success_times['cylinder_10m'] is None and horizontal_distance <= 10.0:
            self.criteria_success_times['cylinder_10m'] = current_time
            self.criteria_success_distances['cylinder_10m'] = horizontal_distance
            logger.info(f"[Gym-Eval-{self.instance_id}] cylinder_10m SUCCESS at {current_time:.2f}s, horizontal={horizontal_distance:.2f}m")

        if self.criteria_success_times['cylinder_10m'] is not None:
            terminated = True
            reward += SUCCESS_RADIUS_BONUS
            info = {'success': True, 'reason': 'success'}
            logger.info(f"[Gym-Eval-{self.instance_id}] Episode ended: cylinder_10m success")
        elif time_norm > 1.0:
            truncated, info = True, {'success': False, 'reason': 'timeout'}
            reward += TRUNCATED_PENALTY
            logger.warning(f"[Gym-Eval-{self.instance_id}] Episode ended: timeout")
        elif not (0 <= alt_limit <= 1):
            truncated, info = True, {'success': False, 'reason': 'altitude'}
            reward += TRUNCATED_PENALTY
            logger.warning(f"[Gym-Eval-{self.instance_id}] Episode ended: altitude violation")
        elif geo_fence > 1:
            truncated, info = True, {'success': False, 'reason': 'distance'}
            reward += TRUNCATED_PENALTY
            logger.warning(f"[Gym-Eval-{self.instance_id}] Episode ended: geofence violation")

        return reward, terminated, truncated, info

    def normalize_angle(self, angle):
        """Radians → [-1, +1]."""
        return ((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi

    def close(self):
        """Stop the simulators and flush the metrics file."""
        logger.info(f"[Gym-Eval-{self.instance_id}] Closing environment and stopping simulators.")
        self.px4.stop()
        self.jmavsim.stop()
        self._flush_pending_ulog()

        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            logger.info(f"[Gym-Eval-{self.instance_id}] Saving final metrics...")
            self.metrics_collector.save(print_summary=True)

        super().close()


    def apply_sensor_noise(self, arr: np.ndarray) -> np.ndarray:
        """Return GT hil_state with 1 Hz reference noise applied."""
        noisy = arr.copy()

        noisy += np.random.normal(0.0, self.sensor_noise_std, 
                                size=noisy.shape).astype(np.float32)

        euler_noise = np.deg2rad(np.random.normal(0.0, self.attitude_noise_std_deg))
        q_orig = noisy[9:13]
        rot_orig = Rotation.from_quat([q_orig[1], q_orig[2], q_orig[3], q_orig[0]])
        rot_noise = Rotation.from_euler('xyz', euler_noise)
        rot_noisy = (rot_noise * rot_orig).as_quat()
        noisy[9:13] = np.array([rot_noisy[3], rot_noisy[0], 
                                rot_noisy[1], rot_noisy[2]], dtype=np.float32)
        
        return np.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)
