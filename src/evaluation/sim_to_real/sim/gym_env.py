"""Sim-to-real-model SITL eval env for PX4-jMAVSim."""

import math
import os
import random
import time

import gymnasium as gym
import numpy as np
from geographiclib.geodesic import Geodesic

from config import (
    ACT_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    DISTANCE_K,
    EP_TIMEOUT,
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
from px4_controller_multi_topic import PX4MultiTopicController

from evaluation.common.metrics import MetricsCollector, TestResult

from evaluation.sim_to_real.common.observation import (
    process_observation as _process_observation_fn,
    build_dict_observation as _build_dict_observation_fn,
)


class PX4RLEnvSimToRealEval(gym.Env):
    """Multi-criterion SITL eval env for the sim-to-real model."""

    def _flush_pending_ulog(self):
        if hasattr(self, "_pending_ulog_outcome"):
            success, ep = self._pending_ulog_outcome
            self.px4.organize_ulog_files(
                success=success,
                episode_num=ep,
                experiment="gap-sim-to-real",
                platform="px4-jmavsim",
            )
            del self._pending_ulog_outcome

    def __init__(self, env_config):
        super().__init__()

        self.time_penalty = env_config.get("TIME_PENALTY", TIME_PENALTY)
        self.distance_k = env_config.get("DISTANCE_K", DISTANCE_K)
        self.velocity_k = env_config.get("VELOCITY_K", VELOCITY_K)

        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(ACTOR_OBS_DIM,), dtype=np.float32,
            ),
            "critic": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(CRITIC_OBS_DIM,), dtype=np.float32,
            ),
        })
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32,
        )

        if hasattr(env_config, "worker_index"):
            self.instance_id = env_config.worker_index
        elif isinstance(env_config, dict) and "worker_index" in env_config:
            self.instance_id = env_config["worker_index"]
        else:
            self.instance_id = 0

        logger.info(
            f"[SimToReal-Eval-{self.instance_id}] init: worker_idx={self.instance_id}"
        )

        _log_dir = (
            os.environ.get("GAP_RESULTS_DIR")
            or os.environ.get("GAP_LOG_DIR")
            or "."
        )
        self.metrics_collector = MetricsCollector(
            output_dir=_log_dir,
            experiment="gap-sim-to-real",
            platform="px4-jmavsim",
            worker=self.instance_id,
        )
        self.milestone_interval = 10
        self.test_name_base = f"gap_sim_to_real_px4-jmavsim_w{self.instance_id:02d}"
        logger.info(
            f"[SimToReal-Eval-{self.instance_id}] Metrics: {self.metrics_collector.log_file}"
        )

        sp_list = env_config.get("start_points", None)
        if sp_list:
            idx = self.instance_id - 1 if 1 <= self.instance_id <= len(sp_list) else 0
            pt = sp_list[idx]
            self.fixed_start_lat = pt["lat"]
            self.fixed_start_lon = pt["lon"]
        else:
            self.fixed_start_lat = None
            self.fixed_start_lon = None

        self.px4 = PX4MultiTopicController(
            instance_id=self.instance_id,
            speed_factor=PX4_PARAM_SPEED,
            headless=PX4_PARAM_HEADLESS,
        )
        self.jmavsim = JmavsimController(
            instance_id=self.instance_id,
            speed_factor=PX4_PARAM_SPEED,
            headless=PX4_PARAM_HEADLESS,
        )

        self.initial_hil = np.zeros(16, np.float32)
        self.episode_start_time = 0.0
        self.previous_sub_step_distance_to_target_2d = None

        self.step_times = []
        self.step_times_current = []
        self.last_log_time = None
        self.step_end_time = None
        self.step_count = 0

        self.criteria_success_times = {
            "sphere_20m": None, "cylinder_20m": None,
            "sphere_10m": None, "cylinder_10m": None,
        }
        self.criteria_success_distances = {
            "sphere_20m": None, "cylinder_20m": None,
            "sphere_10m": None, "cylinder_10m": None,
        }

    def reset(self, *, seed=None, options=None, max_retries=5):
        super().reset(seed=seed)
        logger.debug(f"[SimToReal-Eval-{self.instance_id}] Reset called.")
        self._episode_counter = getattr(self, "_episode_counter", 0)

        self.criteria_success_times = {k: None for k in self.criteria_success_times}
        self.criteria_success_distances = {k: None for k in self.criteria_success_distances}

        for attempt in range(max_retries):
            try:
                self.px4.stop()
                self.jmavsim.stop()

                # Export the previous ulog only after PX4 has flushed it.
                if attempt == 0 and hasattr(self, "_pending_ulog_outcome"):
                    self._flush_pending_ulog()

                start_lat, start_lon = None, None
                if options and "start_lat" in options and "start_lon" in options:
                    start_lat, start_lon = options["start_lat"], options["start_lon"]
                elif self.fixed_start_lat is not None:
                    start_lat, start_lon = self.fixed_start_lat, self.fixed_start_lon
                else:
                    angle_rad = random.uniform(0, 2 * math.pi)
                    offset_north = RANDOM_START_RADIUS_M * math.cos(angle_rad)
                    offset_east = RANDOM_START_RADIUS_M * math.sin(angle_rad)
                    m_per_deg_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(TARGET_LAT))
                    start_lat = TARGET_LAT + offset_north / METERS_PER_DEGREE_LAT
                    start_lon = TARGET_LON + offset_east / m_per_deg_lon

                self.jmavsim.start(latitude=start_lat, longitude=start_lon)
                self.jmavsim.wait_for_ready(timeout=60)
                self.px4.start_px4(latitude=start_lat, longitude=start_lon)

                if not self.px4.wait_px4_ready(timeout=60):
                    raise RuntimeError("PX4 startup failed")
                self.px4.set_gyro_bias(bias=ZERO_ACTION)

                health_ok, _ = self.px4.wait_health_ok(timeout=60)
                if not health_ok:
                    raise RuntimeError("PX4 health check failed")
                self.px4.set_gyro_bias(bias=ZERO_ACTION)

                time.sleep(10)
                self.px4.set_gyro_bias(bias=ZERO_ACTION)

                self.px4.drain_all()
                hil_state = self.px4.get_hil(timeout=STEP_BASE)
                if hil_state is None:
                    raise RuntimeError("Failed to receive pre-takeoff HIL data")
                self.initial_hil = hil_state.copy()

                if not self.px4.arm_and_takeoff(
                    hover_time=60, target_alt_m=self.initial_hil[2] + TAKEOFF_ALT,
                ):
                    raise RuntimeError("Takeoff failed")

                self.episode_start_time = time.monotonic()

                self.px4.drain_all()
                hil_state = self.px4.get_hil(timeout=STEP_BASE)
                if hil_state is None:
                    raise RuntimeError("Failed to receive post-takeoff HIL data")

                processed_hil_state = self.process_observation(hil_state).copy()
                processed_privileged_info = np.zeros(PRIVILEGED_INFO_DIM, dtype=np.float32)
                self.previous_sub_step_distance_to_target_2d = processed_hil_state[16]

                logger.info(
                    f"[SimToReal-Eval-{self.instance_id}] Environment reset complete."
                )

                self.step_times.clear()
                self.step_times_current.clear()
                self.last_log_time = time.monotonic()
                self.step_end_time = time.monotonic()
                self.step_count = 0

                return (
                    self._build_dict_observation(
                        processed_hil_state, processed_privileged_info, ZERO_ACTION,
                    ),
                    {},
                )

            except Exception as e:
                logger.debug(
                    f"[SimToReal-Eval-{self.instance_id}] Reset attempt "
                    f"{attempt+1}/{max_retries} failed: {e}. Retrying..."
                )
                time.sleep(3)

        logger.critical(
            f"[SimToReal-Eval-{self.instance_id}] Reset failed after all retries."
        )
        self.step_times.clear()
        self.step_times_current.clear()
        self.last_log_time = time.monotonic()
        self.step_end_time = time.monotonic()
        self.step_count = 0

        dummy_actor = np.zeros(self.observation_space["actor"].shape, dtype=np.float32)
        dummy_critic = np.zeros(self.observation_space["critic"].shape, dtype=np.float32)
        return (
            {"actor": dummy_actor, "critic": dummy_critic},
            {"error": "reset_failed_after_retries"},
        )

    def step(self, action):
        try:
            total_reward = 0.0
            terminated = truncated = False
            self.step_count += 1

            step_start_time = time_previous = time.monotonic()
            bias = np.array([action[0] * MAX_BIAS, action[1] * MAX_BIAS, 0.0])
            self.px4.set_gyro_bias(bias=bias)

            if self.step_count % 10 == 0:
                logger.info(
                    f"[{self.test_name_base}] Step {self.step_count}: "
                    f"action=[{action[0]:.3f}, {action[1]:.3f}], "
                    f"gyro_bias=[{bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}] rad/s"
                )

            self.px4.drain_all()

            hil_state = None
            processed_hil_state = None
            info = {}
            while time.monotonic() - step_start_time < STEP_BASE:
                now = time.monotonic()
                delta_time = now - time_previous
                time_previous = now

                sub_hil = self.px4.get_hil(timeout=MAV_TIMEOUT)
                if sub_hil is None:
                    continue

                hil_state = sub_hil
                processed_hil_state = self.process_observation(hil_state).copy()

                reward, terminated, truncated, info = self._calculate_reward(
                    processed_hil_state, delta_time=delta_time,
                )
                self.previous_sub_step_distance_to_target_2d = processed_hil_state[16]
                total_reward += reward

                if terminated or truncated:
                    break

            if hil_state is None:
                logger.warning(
                    f"[SimToReal-Eval-{self.instance_id}] No HIL in step; truncating"
                )
                truncated = True
                info = {"reason": "step_hil_timeout"}
                hil_state = self.initial_hil
                processed_hil_state = self.process_observation(hil_state).copy()

            processed_privileged_info = np.zeros(PRIVILEGED_INFO_DIM, dtype=np.float32)

            current_gyro_bias = self.px4.get_gyro_bias(timeout=MAV_TIMEOUT)
            if not terminated and not truncated:
                if current_gyro_bias is None:
                    raise RuntimeError("Failed to receive STEP gyro bias data")
                if not np.allclose(current_gyro_bias, bias, atol=1e-4):
                    logger.warning(
                        f"[SimToReal-Eval-{self.instance_id}] Gyro Bias mismatch: "
                        f"expected {bias}, current {current_gyro_bias}"
                    )
                    raise RuntimeError(
                        f"Gyro Bias mismatch: expected {bias}, current {current_gyro_bias}"
                    )

            dict_obs = self._build_dict_observation(
                processed_hil_state, processed_privileged_info, bias,
            )

            step_duration = time.monotonic() - self.step_end_time
            self.step_times.append(step_duration)
            self.step_end_time = time.monotonic()
            self.step_times_current.append(self.step_end_time - step_start_time)

            if terminated or truncated:
                ep = getattr(self, "_episode_counter", 0) + 1
                self._episode_counter = ep

                time_spent = time.monotonic() - self.episode_start_time
                test_id = len(self.metrics_collector.results) + 1
                result = TestResult(test_id, self.instance_id, ep)
                result.time_spent_s = time_spent
                result.attack_steps = self.step_count

                result.final_distance_m = float(processed_hil_state[17])

                latitude_position, longitude_position, altitude_amsl = hil_state[0:3]
                geodesic = Geodesic.WGS84.Inverse(
                    latitude_position, longitude_position, TARGET_LAT, TARGET_LON,
                )
                horizontal_distance = geodesic["s12"]
                relative_altitude = altitude_amsl - self.initial_hil[2]
                vertical_distance = relative_altitude - TAKEOFF_ALT
                result.horizontal_distance_m = float(horizontal_distance)
                result.vertical_distance_m = float(vertical_distance)
                result.final_altitude_amsl = float(altitude_amsl)
                result.final_relative_altitude = float(relative_altitude)

                result.sphere_20m_success = self.criteria_success_times["sphere_20m"] is not None
                result.sphere_20m_time_s = self.criteria_success_times["sphere_20m"]
                result.sphere_20m_distance_m = self.criteria_success_distances["sphere_20m"]

                result.cylinder_20m_success = self.criteria_success_times["cylinder_20m"] is not None
                result.cylinder_20m_time_s = self.criteria_success_times["cylinder_20m"]
                result.cylinder_20m_horizontal_m = self.criteria_success_distances["cylinder_20m"]

                result.sphere_10m_success = self.criteria_success_times["sphere_10m"] is not None
                result.sphere_10m_time_s = self.criteria_success_times["sphere_10m"]
                result.sphere_10m_distance_m = self.criteria_success_distances["sphere_10m"]

                result.cylinder_10m_success = self.criteria_success_times["cylinder_10m"] is not None
                result.cylinder_10m_time_s = self.criteria_success_times["cylinder_10m"]
                result.cylinder_10m_horizontal_m = self.criteria_success_distances["cylinder_10m"]
                if result.cylinder_10m_success:
                    result.terminal_reason = "success"
                elif isinstance(info, dict):
                    result.terminal_reason = info.get("reason", None)
                else:
                    result.terminal_reason = "truncated" if truncated else None

                self.metrics_collector.add_result(result.to_dict())
                self.metrics_collector.save(print_summary=False)

                self._pending_ulog_outcome = (bool(result.cylinder_10m_success), ep)

                achieved = {
                    "sphere_20m": result.sphere_20m_success,
                    "cylinder_20m": result.cylinder_20m_success,
                    "sphere_10m": result.sphere_10m_success,
                    "cylinder_10m": result.cylinder_10m_success,
                }
                logger.info(
                    f"[{self.test_name_base}] Test {test_id} done. Criteria: "
                    + ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in achieved.items())
                )

                if ep % self.milestone_interval == 0:
                    self.metrics_collector.print_summary()

            return dict_obs, total_reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"[SimToReal-Eval-{self.instance_id}] step error", exc_info=True)
            dummy_actor = np.zeros(self.observation_space["actor"].shape, dtype=np.float32)
            dummy_critic = np.zeros(self.observation_space["critic"].shape, dtype=np.float32)
            return (
                {"actor": dummy_actor, "critic": dummy_critic},
                0, False, True, {"error": repr(e)},
            )

    def process_observation(self, hil_state):
        """Delegate to the shared sim-to-real observation helper."""
        return _process_observation_fn(
            hil_state, self.initial_hil, self.episode_start_time,
            target_lat=TARGET_LAT, target_lon=TARGET_LON,
            takeoff_alt=TAKEOFF_ALT, ep_timeout=EP_TIMEOUT,
            success_radius_m=SUCCESS_RADIUS_M,
            safe_alt_min=SAFE_ALT_MIN, safe_alt_max=SAFE_ALT_MAX,
            safe_target_dist=SAFE_TARGET_DIST,
        )

    def _build_dict_observation(self, processed_hil, processed_priv, action):
        return _build_dict_observation_fn(
            processed_hil, processed_priv, action,
            ACTOR_OBS_DIM, CRITIC_OBS_DIM,
        )

    def _calculate_reward(self, obs, delta_time=None):
        """Track all four success criteria and end once all are achieved."""
        time_norm = obs[0]
        distance_2d = obs[16]
        distance_3d = obs[17]
        closing_vel = obs[18]
        alt_limit = obs[21]
        geo_fence = obs[22]

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if delta_time is None:
            delta_time = STEP_BASE

        if self.previous_sub_step_distance_to_target_2d is not None:
            progress = self.previous_sub_step_distance_to_target_2d - distance_2d
            reward += progress * self.distance_k

        closing_vel_clipped = np.clip(closing_vel, -5.0, 5.0)
        reward += closing_vel_clipped * self.velocity_k * (delta_time / STEP_BASE)
        reward += self.time_penalty * (delta_time / STEP_BASE)

        current_time = time.monotonic() - self.episode_start_time

        if self.criteria_success_times["sphere_20m"] is None and distance_3d <= 20.0:
            self.criteria_success_times["sphere_20m"] = current_time
            self.criteria_success_distances["sphere_20m"] = distance_3d
            logger.info(
                f"[{self.test_name_base}] ✓ sphere_20m at step {self.step_count}: "
                f"2D={distance_2d:.2f}m, 3D={distance_3d:.2f}m, t={current_time:.2f}s"
            )
        if self.criteria_success_times["sphere_10m"] is None and distance_3d <= 10.0:
            self.criteria_success_times["sphere_10m"] = current_time
            self.criteria_success_distances["sphere_10m"] = distance_3d
            logger.info(
                f"[{self.test_name_base}] ✓ sphere_10m at step {self.step_count}: "
                f"2D={distance_2d:.2f}m, 3D={distance_3d:.2f}m, t={current_time:.2f}s"
            )
        if self.criteria_success_times["cylinder_20m"] is None and distance_2d <= 20.0:
            self.criteria_success_times["cylinder_20m"] = current_time
            self.criteria_success_distances["cylinder_20m"] = distance_2d
            logger.info(
                f"[{self.test_name_base}] ✓ cylinder_20m at step {self.step_count}: "
                f"2D={distance_2d:.2f}m, 3D={distance_3d:.2f}m, t={current_time:.2f}s"
            )
        if self.criteria_success_times["cylinder_10m"] is None and distance_2d <= 10.0:
            self.criteria_success_times["cylinder_10m"] = current_time
            self.criteria_success_distances["cylinder_10m"] = distance_2d
            logger.info(
                f"[{self.test_name_base}] ✓ cylinder_10m at step {self.step_count}: "
                f"2D={distance_2d:.2f}m, 3D={distance_3d:.2f}m, t={current_time:.2f}s"
            )

        if all(v is not None for v in self.criteria_success_times.values()):
            terminated = True
            reward += SUCCESS_RADIUS_BONUS
            info = {"success": True, "reason": "success"}
            logger.info(f"[{self.test_name_base}] All 4 success criteria achieved! Early termination.")

        if terminated:
            return reward, terminated, truncated, info

        if time_norm > 1.0:
            truncated, info = True, {"success": False, "reason": "timeout"}
            reward += TRUNCATED_PENALTY
            logger.warning(f"[{self.test_name_base}] Episode ended: timeout")
        elif not (0 <= alt_limit <= 1):
            truncated, info = True, {"success": False, "reason": "altitude"}
            reward += TRUNCATED_PENALTY
            logger.warning(f"[{self.test_name_base}] Episode ended: altitude")
        elif geo_fence > 1:
            truncated, info = True, {"success": False, "reason": "distance"}
            reward += TRUNCATED_PENALTY
            logger.warning(f"[{self.test_name_base}] Episode ended: geofence")

        return reward, terminated, truncated, info

    def normalize_angle(self, angle):
        return ((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi

    def close(self):
        logger.info(f"[SimToReal-Eval-{self.instance_id}] Closing env.")
        self.px4.stop()
        self.jmavsim.stop()
        self._flush_pending_ulog()
        if hasattr(self, "metrics_collector") and self.metrics_collector:
            self.metrics_collector.save(print_summary=True)
