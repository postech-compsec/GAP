"""Run GAP on PX4 Gazebo and ArduPilot frames without retraining."""

import time
import argparse
from pathlib import Path
import numpy as np
import math
from geographiclib.geodesic import Geodesic
from scipy.spatial.transform import Rotation

from evaluation.common import (
    MetricsCollector, TestResult,
    TARGET_LAT, TARGET_LON, TAKEOFF_ALT,
    DRONE_START_LAT, DRONE_START_LON,
    SIMULATION_SPEEDUP, TEST_TIMEOUT_SIM_SEC,
    SUCCESS_RADIUS_M,
    get_start_positions, get_target_positions, check_success, logger, LOG_DIR,
    MAX_GYRO_BIAS, NUM_START_POSITIONS, NUM_TRIALS_PER_POSITION,
    METERS_PER_DEGREE_LAT, ARDUPILOT_FRAMES,
    GAP_MODEL_PATH,
)
from evaluation.common.ray_logging import make_logger_creator
from evaluation.common.controllers import PX4GazeboController, ArdupilotController

import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from gap.asymmetric_rl_module import AsymmetricLSTMModule


class ObservationProcessor:
    """Map raw simulator state into the actor observation."""

    def __init__(self, initial_hil_state, episode_start_time, target_lat, target_lon,
                 simulation_speedup: int = 1):
        self.initial_hil = initial_hil_state
        self.episode_start_time = episode_start_time
        self.target_lat = target_lat
        self.target_lon = target_lon
        self.simulation_speedup = simulation_speedup

    def process_observation(self, hil_state):
        """Return the 21-D actor observation used by the trained policy."""
        latitude, longitude, altitude_amsl = hil_state[0:3]
        vx, vy, vz = hil_state[3:6]
        ax, ay, az = hil_state[6:9]
        quaternion_wxyz = hil_state[9:13]
        rollspeed, pitchspeed, yawspeed = hil_state[13:16]

        time_since_reset = time.monotonic() - self.episode_start_time
        normalized_time = (time_since_reset * self.simulation_speedup) / TEST_TIMEOUT_SIM_SEC

        geodesic = Geodesic.WGS84.Inverse(latitude, longitude, self.target_lat, self.target_lon)
        direction_to_target_ned = np.deg2rad(geodesic['azi1'])
        distance_to_target = geodesic['s12']
        north_offset = distance_to_target * np.cos(direction_to_target_ned)
        east_offset = distance_to_target * np.sin(direction_to_target_ned)

        relative_altitude = altitude_amsl - self.initial_hil[2]
        z_to_target = relative_altitude - TAKEOFF_ALT
        relative_position_ned = np.array([north_offset, east_offset, z_to_target], np.float32)

        rotation_frd_to_ned = Rotation.from_quat(quaternion_wxyz, scalar_first=True)
        relative_position_frd = rotation_frd_to_ned.apply(relative_position_ned, inverse=True)
        velocity_ned = np.array([vx, vy, vz], np.float32)
        velocity_frd = rotation_frd_to_ned.apply(velocity_ned, inverse=True)

        roll, pitch, yaw = rotation_frd_to_ned.as_euler('xyz', degrees=False)

        def normalize_angle(angle):
            return ((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi

        normalized_roll = normalize_angle(roll)
        normalized_pitch = normalize_angle(pitch)
        normalized_yaw = normalize_angle(yaw)
        heading_error = normalize_angle(direction_to_target_ned - yaw)
        normalized_direction = normalize_angle(direction_to_target_ned)

        distance_3d = float(np.sqrt(distance_to_target**2 + z_to_target**2))

        if distance_3d > 1e-6:
            target_unit_vec = relative_position_ned / distance_3d
            closing_velocity = np.dot(velocity_ned, target_unit_vec)
        else:
            closing_velocity = 0.0

        SAFE_ALT_MIN = 0.0
        SAFE_ALT_MAX = 120.0
        distance_to_alt_limit = (relative_altitude - SAFE_ALT_MIN) / (SAFE_ALT_MAX - SAFE_ALT_MIN)

        SAFE_TARGET_DIST = 270.0
        distance_to_geofence = distance_to_target / SAFE_TARGET_DIST

        is_in_sphere = 1 if distance_to_target <= SUCCESS_RADIUS_M else 0

        total_velocity = np.linalg.norm(velocity_frd)

        obs = np.array([
            normalized_time,
            *relative_position_frd,  # 3
            *velocity_frd,  # 3
            normalized_roll, normalized_pitch, normalized_yaw,  # 3
            rollspeed, pitchspeed, yawspeed,  # 3
            distance_3d,  # 1
            closing_velocity,  # 1
            normalized_direction,  # 1
            heading_error,  # 1
            distance_to_alt_limit,  # 1
            distance_to_geofence,  # 1
            is_in_sphere,  # 1
            total_velocity,  # 1
        ], dtype=np.float32)

        return obs


class RLTest:
    """Run GAP on PX4 Gazebo or an ArduPilot frame."""

    def __init__(self, controller, platform: str, frame: str, checkpoint_path: str, speedup: int = None):
        self.controller = controller
        self.platform = platform
        self.frame = frame
        self.test_name_base = f"gap_{platform}_{frame}"
        self.checkpoint_path = checkpoint_path
        self.simulation_speedup = speedup if speedup is not None else SIMULATION_SPEEDUP

        self.all_criteria = ["sphere_20m", "cylinder_20m", "sphere_10m", "cylinder_10m"]
        self.metrics = MetricsCollector(
            output_dir=LOG_DIR,
            experiment="gap",
            platform=platform,
            frame=frame,
        )

        self.rl_model = self.load_rl_model(checkpoint_path)

    def load_rl_model(self, checkpoint_path):
        """Restore the trained GAP policy."""
        checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        logger.info(f"[{self.test_name_base}] Loading RL model from: {checkpoint_path}")

        ray.init(ignore_reinit_error=True, num_gpus=0)

        import gymnasium as gym

        class DummyEnv(gym.Env):
            def __init__(self, config=None):
                super().__init__()
                self.observation_space = gym.spaces.Dict({
                    "actor": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32),
                    "critic": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(79,), dtype=np.float32),
                })
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return {
                    "actor": np.zeros(24, dtype=np.float32),
                    "critic": np.zeros(79, dtype=np.float32)
                }, {}

            def step(self, action):
                obs = {
                    "actor": np.zeros(24, dtype=np.float32),
                    "critic": np.zeros(79, dtype=np.float32)
                }
                return obs, 0.0, False, False, {}

        from ray.tune.registry import register_env
        register_env("dummy-env", lambda cfg: DummyEnv(cfg))

        config = (
            APPOConfig()
            .environment("dummy-env", env_config={})
            .framework("torch")
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .env_runners(num_env_runners=0)
            .rl_module(
                rl_module_spec=RLModuleSpec(module_class=AsymmetricLSTMModule),
            )
            .evaluation(
                evaluation_config={
                    "explore": False,
                }
            )
        )

        algo = config.build_algo(
            logger_creator=make_logger_creator(f"rq3_{self.platform}_{self.frame}")
        )
        algo.restore(checkpoint_path)

        logger.info(f"[{self.test_name_base}] RL model loaded successfully")
        return algo

    def predict_action(self, observation, prev_action, lstm_state=None):
        """Run one inference step and return the next action and state."""
        if self.rl_model is None:
            logger.error(f"[{self.test_name_base}] RL model not loaded")
            return np.zeros(2, dtype=np.float32), lstm_state

        actor_obs_with_action = np.concatenate([observation, prev_action], dtype=np.float32)
        rl_module = self.rl_model.get_module()

        if lstm_state is None:
            lstm_state = rl_module.get_initial_state()

        import torch
        from ray.rllib.core.columns import Columns
        obs_batch = {
            Columns.OBS: {
                "actor": torch.from_numpy(actor_obs_with_action).unsqueeze(0),
            }
        }

        state_batch = {}
        if lstm_state:
            for key, value in lstm_state.items():
                if isinstance(value, np.ndarray):
                    state_batch[key] = torch.from_numpy(value)
                else:
                    state_batch[key] = value

        if state_batch:
            obs_batch[Columns.STATE_IN] = state_batch

        with torch.no_grad():
            output = rl_module.forward_inference(obs_batch)

        action_dist_inputs = output.get(Columns.ACTION_DIST_INPUTS, None)
        if action_dist_inputs is not None:
            action = action_dist_inputs.squeeze().cpu().numpy()[:2]
        else:
            logger.error(f"[{self.test_name_base}] No action_dist_inputs in model output")
            action = np.zeros(2, dtype=np.float32)

        new_state = output.get(Columns.STATE_OUT, {})
        new_state_dict = {}
        for key, value in new_state.items():
            if torch.is_tensor(value):
                new_state_dict[key] = value.cpu().numpy()
            else:
                new_state_dict[key] = value

        return action, new_state_dict if new_state_dict else lstm_state

    def initialize_drone(self, start_lat: float, start_lon: float, max_retries: int = 3):
        """Start the simulator, take off, and wait for a stable hover."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[{self.test_name_base}] Initialization attempt {attempt}/{max_retries}")

                self.controller.stop()
                time.sleep(2)

                self.controller.start(latitude=start_lat, longitude=start_lon)

                if not self.controller.wait_ready(timeout=60):
                    raise RuntimeError("Failed to get heartbeat")

                self.controller.set_gyro_bias(np.zeros(3, dtype=np.float32))
                time.sleep(10 / SIMULATION_SPEEDUP)

                health_ok, home_coords = self.controller.wait_health_ok()
                if not health_ok:
                    raise RuntimeError("Health check failed")

                self.controller.set_gyro_bias(np.zeros(3, dtype=np.float32))
                time.sleep(10 / SIMULATION_SPEEDUP)
                self.controller.set_gyro_bias(np.zeros(3, dtype=np.float32))

                self.controller.drain_all()
                initial_state = self.get_state(timeout=1.0)
                if initial_state is None:
                    raise RuntimeError("Failed to get initial state")

                logger.info(f"[{self.test_name_base}] Initial position: lat={initial_state[0]:.6f}, lon={initial_state[1]:.6f}")

                target_alt = TAKEOFF_ALT
                if not self.controller.arm_and_takeoff(target_alt_m=target_alt, hover_time=60):
                    raise RuntimeError("Failed to takeoff")

                stabilize_wall_s = 60 / SIMULATION_SPEEDUP
                logger.info(
                    f"[{self.test_name_base}] Stabilizing hover for "
                    f"{stabilize_wall_s:.1f}s wall-clock (60s sim)"
                )
                time.sleep(stabilize_wall_s)
                self.controller.drain_all()
                hover_state = self.get_state(timeout=1.0)
                if hover_state is None:
                    raise RuntimeError("Failed to get hover state")

                logger.info(
                    f"[{self.test_name_base}] Hover stabilized at altitude: "
                    f"{hover_state[2]:.2f}m"
                )

                return True, home_coords, initial_state

            except Exception as e:
                logger.error(f"[{self.test_name_base}] Initialization attempt {attempt} failed: {e}")
                self.controller.stop()
                time.sleep(3)

        logger.error(f"[{self.test_name_base}] Failed to initialize after {max_retries} attempts")
        return False, None, None

    def get_state(self, timeout: float = 1.0):
        """Get the platform-specific state vector."""
        if hasattr(self.controller, 'get_hil'):
            return self.controller.get_hil(timeout)
        elif hasattr(self.controller, 'get_sim_state'):
            return self.controller.get_sim_state(timeout)
        else:
            raise NotImplementedError("Controller does not support state retrieval")

    def confirm_gyro_bias(self, expected_bias: np.ndarray):
        """Verify ArduPilot applied the commanded bias."""
        if self.platform != "ardupilot":
            return

        confirmed_bias = self.controller.get_gyro_bias(
            timeout=max(0.25, 2.0 / self.simulation_speedup)
        )
        if confirmed_bias is None:
            raise RuntimeError("Failed to confirm ArduPilot gyro bias")
        if not np.allclose(confirmed_bias, expected_bias, atol=1e-4):
            raise RuntimeError(
                f"Gyro bias mismatch: expected {expected_bias}, confirmed {confirmed_bias}"
            )

    def run_test_episode(self, position_index: int, trial_number: int):
        """Run one RL episode; return ONE multi-criterion TestResult."""
        test_id = position_index * NUM_TRIALS_PER_POSITION + trial_number
        result = TestResult(test_id, position_index, trial_number)
        achieved = {criterion: False for criterion in self.all_criteria}

        # Get target positions (drone stays at center, targets move around)
        target_positions = get_target_positions(DRONE_START_LAT, DRONE_START_LON, NUM_START_POSITIONS, 220.0)
        target_pos = target_positions[position_index]

        logger.info(f"\n{'='*80}")
        logger.info(f"[{self.test_name_base}] Test {test_id}/{NUM_START_POSITIONS * NUM_TRIALS_PER_POSITION}: "
                   f"Position {position_index} (Clock {target_pos['clock_position']}), Trial {trial_number}")
        logger.info(f"[{self.test_name_base}] Drone at: LAT={target_pos['drone_lat']:.6f}, LON={target_pos['drone_lon']:.6f}")
        logger.info(f"[{self.test_name_base}] Target at: LAT={target_pos['target_lat']:.6f}, LON={target_pos['target_lon']:.6f}")
        logger.info(f"[{self.test_name_base}] Monitoring all 4 success criteria simultaneously")
        logger.info(f"{'='*80}")

        success, home_coords, initial_hil = self.initialize_drone(target_pos['drone_lat'], target_pos['drone_lon'])
        if not success:
            result.terminal_reason = "initialization_failed"
            return result

        episode_start_time = time.monotonic()
        obs_processor = ObservationProcessor(
            initial_hil,
            episode_start_time,
            target_pos['target_lat'],
            target_pos['target_lon'],
            simulation_speedup=self.simulation_speedup,
        )

        attack_step = 0
        if self.rl_model:
            rl_module = self.rl_model.get_module()
            lstm_state = rl_module.get_initial_state()
        else:
            lstm_state = None
        prev_action = np.zeros(3, dtype=np.float32)

        try:
            if hasattr(self.controller, 'clear_failsafe_events'):
                self.controller.clear_failsafe_events()

            logger.info(f"[{self.test_name_base}] Starting RL inference loop")
            while True:
                step_start_time = time.monotonic()

                elapsed_time = step_start_time - episode_start_time
                sim_time = elapsed_time * self.simulation_speedup
                if sim_time >= TEST_TIMEOUT_SIM_SEC:
                    if not all(achieved.values()):
                        result.terminal_reason = "timeout"
                    logger.warning(f"[{self.test_name_base}] Test timed out after {sim_time:.1f}s")
                    break

                self.controller.drain_all()
                current_state = self.get_state(timeout=1.0)
                if current_state is None:
                    if not all(achieved.values()):
                        result.terminal_reason = "lost_connection"
                    logger.error(f"[{self.test_name_base}] Lost connection - no state received!")
                    break

                actor_obs = obs_processor.process_observation(current_state)

                action, lstm_state = self.predict_action(actor_obs, prev_action, lstm_state)
                action = np.clip(action, -1.0, 1.0)

                gyro_bias = np.array([
                    action[0] * MAX_GYRO_BIAS,
                    action[1] * MAX_GYRO_BIAS,
                    0.0
                ], dtype=np.float32)

                if attack_step % 10 == 0:
                    logger.info(f"[{self.test_name_base}] Step {attack_step}: action=[{action[0]:.3f}, {action[1]:.3f}], "
                               f"gyro_bias=[{gyro_bias[0]:.4f}, {gyro_bias[1]:.4f}, {gyro_bias[2]:.4f}] rad/s")

                self.controller.set_gyro_bias(gyro_bias)
                attack_step += 1
                prev_action = gyro_bias

                target_alt_amsl = home_coords['alt'] + TAKEOFF_ALT
                step_duration = 1.0 / self.simulation_speedup

                self.controller.drain_all()
                while time.monotonic() - step_start_time < step_duration:
                    if hasattr(self.controller, 'check_failsafe'):
                        self.controller.check_failsafe()

                    current_state = self.get_state(timeout=0.5)
                    if current_state is None:
                        logger.debug(f"[{self.test_name_base}] No state in monitoring loop")
                        continue

                    current_lat, current_lon, current_alt = current_state[0:3]

                    for criterion in self.all_criteria:
                        if not achieved[criterion]:
                            if check_success(current_lat, current_lon, current_alt,
                                           target_pos['target_lat'], target_pos['target_lon'], target_alt_amsl,
                                           criterion):
                                achieved[criterion] = True
                                ach_time = time.monotonic() - episode_start_time
                                geodesic = Geodesic.WGS84.Inverse(current_lat, current_lon,
                                                                 target_pos['target_lat'], target_pos['target_lon'])
                                h_dist = geodesic['s12']
                                v_dist = current_alt - target_alt_amsl
                                d_3d = np.sqrt(h_dist**2 + v_dist**2)
                                result.record_criterion(criterion, ach_time, d_3d, h_dist)
                                logger.info(f"[{self.test_name_base}] ✓ {criterion} at step {attack_step}: "
                                            f"2D={h_dist:.2f}m, 3D={d_3d:.2f}m, t={ach_time:.2f}s")

                self.confirm_gyro_bias(gyro_bias)

                if all(achieved.values()):
                    logger.info(f"[{self.test_name_base}] All 4 success criteria achieved! Early termination.")
                    break

        except Exception as e:
            logger.error(f"[{self.test_name_base}] Episode error: {e}")
            if not all(achieved.values()):
                result.terminal_reason = f"exception: {e}"

        finally:
            final_state = self.get_state(timeout=1.0)
            result.time_spent_s = time.monotonic() - episode_start_time
            result.attack_steps = attack_step

            if final_state is not None:
                final_lat, final_lon, final_alt = final_state[0:3]
                geodesic = Geodesic.WGS84.Inverse(final_lat, final_lon,
                                                 target_pos['target_lat'], target_pos['target_lon'])
                h_dist = geodesic['s12']
                target_alt_amsl = home_coords['alt'] + TAKEOFF_ALT
                v_dist = final_alt - target_alt_amsl
                result.final_distance_m = float(np.sqrt(h_dist**2 + v_dist**2))
                result.horizontal_distance_m = float(h_dist)
                result.vertical_distance_m = float(abs(v_dist))
                result.final_altitude_amsl = float(final_alt)
                result.final_relative_altitude = float(final_alt - home_coords['alt'])

            if result.cylinder_10m_success:
                result.terminal_reason = "success"

            if hasattr(self.controller, 'get_failsafe_events'):
                events = self.controller.get_failsafe_events()
                result.failsafe_occurred = len(events) > 0
                result.failsafe_events = [msg for _, msg in events]

            logger.info(f"[{self.test_name_base}] Test {test_id} done. Criteria: "
                        + ", ".join(f"{c}={'✓' if achieved[c] else '✗'}" for c in self.all_criteria))

            self.controller.stop()
            if hasattr(self.controller, 'copy_flight_log'):
                kwargs = {
                    "success": bool(achieved.get('cylinder_10m')),
                    "experiment": "gap",
                    "platform": self.platform,
                }
                if self.platform == "px4-gazebo":
                    kwargs["frame"] = self.frame
                self.controller.copy_flight_log(position_index, trial_number, **kwargs)
            time.sleep(2)

        return result

    def run_all_tests(self):
        """Run all tests, monitoring all 4 success criteria simultaneously."""
        logger.info(f"\n{'#'*80}\n# {self.test_name_base}: {NUM_START_POSITIONS * NUM_TRIALS_PER_POSITION} flights, 4 criteria per flight\n{'#'*80}\n")
        for position_idx in range(NUM_START_POSITIONS):
            for trial in range(1, NUM_TRIALS_PER_POSITION + 1):
                result = self.run_test_episode(position_idx, trial)
                self.metrics.add_result(result.to_dict())
        log_file = self.metrics.save(print_summary=True)
        logger.info(f"Results saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Cross-platform GAP evaluation")
    parser.add_argument("--platform", type=str, required=True,
                       choices=["px4-gazebo"] + ARDUPILOT_FRAMES,
                       help="Platform to test")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to RL model checkpoint")
    parser.add_argument("--speedup", type=int, default=SIMULATION_SPEEDUP,
                       help="Simulation speedup factor")
    args = parser.parse_args()

    if args.platform == "px4-gazebo":
        controller = PX4GazeboController(instance_id=0, speed_factor=args.speedup, headless=True)
        platform, frame = "px4-gazebo", "x500"
    else:
        controller = ArdupilotController(frame_type=args.platform, speed_factor=args.speedup, headless=True)
        platform, frame = "ardupilot", args.platform

    try:
        test = RLTest(controller, platform, frame, args.checkpoint, speedup=args.speedup)
        test.run_all_tests()

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise SystemExit(1)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
