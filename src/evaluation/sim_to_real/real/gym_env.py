import time

import gymnasium as gym
import numpy as np

from evaluation.sim_to_real.common.observation import (
    build_dict_observation as _build_dict_observation,
    process_observation as _process_observation,
)
from evaluation.sim_to_real.real.config import (
    ACT_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    DISTANCE_K,
    EP_TIMEOUT,
    HIL_STATE_DIM,
    MAV_TIMEOUT,
    MAX_BIAS,
    PRIVILEGED_INFO_DIM,
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
from evaluation.sim_to_real.real.px4_controller_real import PX4RealController as PX4Controller


class PX4RLEnv(gym.Env):
    def __init__(self, env_config):
        super().__init__()

        self.time_penalty = env_config.get("TIME_PENALTY", TIME_PENALTY)
        self.distance_k = env_config.get("DISTANCE_K", DISTANCE_K)
        self.velocity_k = env_config.get("VELOCITY_K", VELOCITY_K)

        self.observation_space = gym.spaces.Dict({
            "actor": gym.spaces.Box(-np.inf, np.inf, (ACTOR_OBS_DIM,), np.float32),
            "critic": gym.spaces.Box(-np.inf, np.inf, (CRITIC_OBS_DIM,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

        self.px4 = PX4Controller()
        self.episode_start_time = 0.0
        self.initial_hil = np.zeros(HIL_STATE_DIM, np.float32)
        self.previous_sub_step_distance_to_target_2d = None
        self.last_hil_state = None

    def reset(self, *, seed=None, options=None, max_retries=5):
        super().reset(seed=seed)
        self.episode_start_time = time.monotonic()

        self.px4.drain_all()
        hil_state = self.px4.get_hil(timeout=MAV_TIMEOUT)
        if hil_state is None:
            raise RuntimeError("Failed to receive HIL data after takeoff")

        # The real path starts after takeoff; shift the reference back to the
        # ground so the processed observation matches the training convention.
        hil_state[2] -= TAKEOFF_ALT
        self.initial_hil = hil_state.copy()
        self.last_hil_state = hil_state.copy()

        processed_hil = self.process_observation(hil_state).copy()
        self.previous_sub_step_distance_to_target_2d = processed_hil[16]

        return self._build_dict_observation(processed_hil, ZERO_ACTION), {}

    def step(self, action):
        try:
            total_reward = 0.0
            terminated = False
            truncated = False
            info = {}
            processed_hil = self.process_observation(self.last_hil_state).copy()

            step_start_time = time_previous = time.monotonic()
            bias = np.array([action[0] * MAX_BIAS, action[1] * MAX_BIAS, 0.0], np.float32)
            self.px4.set_gyro_bias(bias=bias)
            print(
                f"[Gym] Set gyro bias: x:{bias[0]:.4f}, y:{bias[1]:.4f}, z:{bias[2]:.4f}",
                flush=True,
            )

            self.px4.drain_all()

            while time.monotonic() - step_start_time < STEP_BASE:
                now = time.monotonic()
                delta_time = now - time_previous
                time_previous = now

                hil_state = self.px4.get_hil(timeout=MAV_TIMEOUT)
                if hil_state is None:
                    logger.warning("[Gym] HIL data timeout, using previous state")
                    hil_state = self.last_hil_state
                else:
                    self.last_hil_state = hil_state.copy()

                processed_hil = self.process_observation(hil_state).copy()
                reward, terminated, truncated, info = self._calculate_reward(
                    processed_hil, delta_time=delta_time,
                )
                self.previous_sub_step_distance_to_target_2d = processed_hil[16]
                total_reward += reward

                if terminated or truncated:
                    break

            dict_obs = self._build_dict_observation(processed_hil, bias)

            if terminated or truncated:
                zero = np.zeros(3, dtype=np.float32)
                for _ in range(5):
                    self.px4.set_gyro_bias(bias=zero)
                    print("[Gym] Set gyro bias: x:0.0000, y:0.0000, z:0.0000", flush=True)
                    time.sleep(0.1)

            return dict_obs, total_reward, terminated, truncated, info

        except Exception as exc:
            logger.error("[Gym] step error", exc_info=True)
            dummy_actor = np.zeros(self.observation_space["actor"].shape, dtype=np.float32)
            dummy_critic = np.zeros(self.observation_space["critic"].shape, dtype=np.float32)
            return {"actor": dummy_actor, "critic": dummy_critic}, 0.0, False, True, {"error": repr(exc)}

    def process_observation(self, hil_state):
        return _process_observation(
            hil_state,
            self.initial_hil,
            self.episode_start_time,
            target_lat=TARGET_LAT,
            target_lon=TARGET_LON,
            takeoff_alt=TAKEOFF_ALT,
            ep_timeout=EP_TIMEOUT,
            success_radius_m=SUCCESS_RADIUS_M,
            safe_alt_min=SAFE_ALT_MIN,
            safe_alt_max=SAFE_ALT_MAX,
            safe_target_dist=SAFE_TARGET_DIST,
        )

    def _build_dict_observation(self, processed_hil, action):
        processed_privileged = np.zeros(PRIVILEGED_INFO_DIM, dtype=np.float32)
        return _build_dict_observation(
            processed_hil,
            processed_privileged,
            action,
            ACTOR_OBS_DIM,
            CRITIC_OBS_DIM,
        )

    def _calculate_reward(self, obs, delta_time=None):
        distance_2d = obs[16]
        distance_3d = obs[17]
        closing_vel = obs[18]
        alt_limit = obs[21]
        geo_fence = obs[22]
        in_sphere = obs[23]
        time_norm = obs[0]

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if delta_time is None:
            delta_time = STEP_BASE

        if self.previous_sub_step_distance_to_target_2d is not None:
            progress = self.previous_sub_step_distance_to_target_2d - distance_2d
            reward += progress * self.distance_k

        reward += np.clip(closing_vel, -5.0, 5.0) * self.velocity_k * (delta_time / STEP_BASE)
        reward += self.time_penalty * (delta_time / STEP_BASE)

        if in_sphere:
            terminated = True
            reward += SUCCESS_RADIUS_BONUS
            info = {"success": True, "final_distance": round(float(distance_2d), 2)}
            logger.warning("[Gym] Mission success: %.1fm", distance_2d)
        elif time_norm > 1.0:
            truncated = True
            reward += TRUNCATED_PENALTY
            info = {"success": False, "reason": "timeout"}
            logger.warning("[Gym] Mission failed: timeout")
        elif not (0 <= alt_limit <= 1):
            truncated = True
            reward += TRUNCATED_PENALTY
            info = {"success": False, "reason": "altitude"}
            logger.warning("[Gym] Mission failed: altitude")
        elif geo_fence > 1:
            truncated = True
            reward += TRUNCATED_PENALTY
            info = {"success": False, "reason": "distance"}
            logger.warning("[Gym] Mission failed: distance")

        return reward, terminated, truncated, info

    def close(self):
        super().close()
