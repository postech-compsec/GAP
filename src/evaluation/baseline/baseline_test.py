"""Non-RL baseline attacks for RQ1."""

import time
import math
import argparse
import numpy as np
from geographiclib.geodesic import Geodesic
from scipy.spatial.transform import Rotation

from evaluation.common import (
    MetricsCollector, TestResult,
    TARGET_LAT, TARGET_LON, TAKEOFF_ALT,
    SIMULATION_SPEEDUP, TEST_TIMEOUT_SIM_SEC,
    get_start_positions, calculate_directional_attack,
    check_success, logger, LOG_DIR, MAX_GYRO_BIAS,
    ATTACK_RANGE_MIN, ATTACK_RANGE_MAX, ATTACK_INTERVAL,
    NUM_START_POSITIONS, NUM_TRIALS_PER_POSITION
)
from evaluation.common.controllers import PX4JMAVSimController


class BaselineTest:
    """Shared baseline runner that records all four criteria per episode."""

    def __init__(self, controller, experiment: str):
        self.controller = controller
        self.test_name_base = experiment
        self.experiment = experiment
        self.all_criteria = ["sphere_20m", "cylinder_20m", "sphere_10m", "cylinder_10m"]
        self.metrics = MetricsCollector(
            output_dir=LOG_DIR,
            experiment=experiment,
            platform="px4-jmavsim",
        )

    def initialize_drone(self, start_lat: float, start_lon: float, max_retries: int = 3):
        """Start SITL, arm, take off to TAKEOFF_ALT. Returns home coords or None."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[{self.test_name_base}] Init attempt {attempt}/{max_retries}")

                self.controller.stop()
                time.sleep(2)
                self.controller.start(latitude=start_lat, longitude=start_lon)
                if not self.controller.wait_ready(timeout=60):
                    raise RuntimeError("Failed to get heartbeat")

                self.controller.set_gyro_bias(np.zeros(3, dtype=np.float32))
                health_ok, home_coords = self.controller.wait_health_ok(timeout=60)
                if not health_ok:
                    raise RuntimeError("Health check failed")

                # PX4 can drop one early zero-bias write during boot.
                self.controller.set_gyro_bias(np.zeros(3, dtype=np.float32))
                time.sleep(10 / SIMULATION_SPEEDUP)
                self.controller.set_gyro_bias(np.zeros(3, dtype=np.float32))

                self.controller.drain_all()
                initial_state = self.controller.get_hil(timeout=1.0)
                if initial_state is None:
                    raise RuntimeError("Failed to get initial HIL state")
                logger.info(f"[{self.test_name_base}] Initial position: "
                            f"lat={initial_state[0]:.6f}, lon={initial_state[1]:.6f}")

                target_alt = initial_state[2] + TAKEOFF_ALT
                if not self.controller.arm_and_takeoff(target_alt_m=target_alt, hover_time=60):
                    raise RuntimeError("Failed to takeoff")

                time.sleep(10 / SIMULATION_SPEEDUP)
                self.controller.drain_all()
                hover_state = self.controller.get_hil(timeout=1.0)
                if hover_state is None:
                    raise RuntimeError("Failed to get hover state")
                logger.info(f"[{self.test_name_base}] Hovering at alt {hover_state[2]:.2f}m")
                return home_coords

            except Exception as e:
                logger.error(f"[{self.test_name_base}] Init attempt {attempt} failed: {e}")
                self.controller.stop()
                time.sleep(3)

        logger.error(f"[{self.test_name_base}] Gave up after {max_retries} attempts")
        return None

    def run_test_episode(self, position_index: int, trial_number: int, attack_func):
        """Run one episode and record all four criteria on one TestResult."""
        test_id = position_index * NUM_TRIALS_PER_POSITION + trial_number
        result = TestResult(test_id, position_index, trial_number)
        achieved = {criterion: False for criterion in self.all_criteria}

        start_positions = get_start_positions(NUM_START_POSITIONS, 220.0)
        start_pos = start_positions[position_index]

        logger.info(f"\n{'='*80}")
        logger.info(f"[{self.test_name_base}] Test {test_id}/{NUM_START_POSITIONS * NUM_TRIALS_PER_POSITION}: "
                   f"Position {position_index} (Clock {start_pos['clock_position']}), Trial {trial_number}")
        logger.info(f"{'='*80}")

        home_coords = self.initialize_drone(start_pos['lat'], start_pos['lon'])
        if home_coords is None:
            result.terminal_reason = "initialization_failed"
            return result

        episode_start_time = time.monotonic()
        attack_step = 0

        try:
            while True:
                step_start_time = time.monotonic()

                elapsed_time = step_start_time - episode_start_time
                sim_time = elapsed_time * SIMULATION_SPEEDUP
                if sim_time >= TEST_TIMEOUT_SIM_SEC:
                    if not all(achieved.values()):
                        result.terminal_reason = "timeout"
                    break

                current_state = self.controller.get_hil(timeout=1.0)
                if current_state is None:
                    if not all(achieved.values()):
                        result.terminal_reason = "lost_connection"
                    break

                current_lat, current_lon, current_alt = current_state[0:3]
                target_alt_amsl = home_coords['alt'] + TAKEOFF_ALT

                for criterion in self.all_criteria:
                    if not achieved[criterion]:
                        if check_success(current_lat, current_lon, current_alt,
                                       TARGET_LAT, TARGET_LON, target_alt_amsl,
                                       criterion):
                            achieved[criterion] = True
                            ach_time = time.monotonic() - episode_start_time
                            geodesic = Geodesic.WGS84.Inverse(current_lat, current_lon,
                                                             TARGET_LAT, TARGET_LON)
                            h_dist = geodesic['s12']
                            v_dist = current_alt - target_alt_amsl
                            d_3d = np.sqrt(h_dist**2 + v_dist**2)
                            result.record_criterion(criterion, ach_time, d_3d, h_dist)
                            logger.info(f"[{self.test_name_base}] ✓ {criterion} at step {attack_step}: "
                                      f"2D={h_dist:.2f}m, 3D={d_3d:.2f}m, t={ach_time:.2f}s")

                if all(achieved.values()):
                    logger.info(f"[{self.test_name_base}] All 4 criteria achieved. Early termination.")
                    break

                attack_bias = attack_func(attack_step, position_index, current_state)
                self.controller.set_gyro_bias(attack_bias)
                attack_step += 1
                time.sleep(ATTACK_INTERVAL / SIMULATION_SPEEDUP)

        except Exception as e:
            logger.error(f"[{self.test_name_base}] Episode error: {e}")
            if not all(achieved.values()):
                result.terminal_reason = f"exception: {e}"

        finally:
            final_state = self.controller.get_hil(timeout=1.0)
            result.time_spent_s = time.monotonic() - episode_start_time
            result.attack_steps = attack_step
            if final_state is not None:
                f_lat, f_lon, f_alt = final_state[0:3]
                geodesic = Geodesic.WGS84.Inverse(f_lat, f_lon, TARGET_LAT, TARGET_LON)
                h_dist = geodesic['s12']
                target_alt_amsl = home_coords['alt'] + TAKEOFF_ALT
                v_dist = f_alt - target_alt_amsl
                result.final_distance_m = float(np.sqrt(h_dist**2 + v_dist**2))
                result.horizontal_distance_m = float(h_dist)
                result.vertical_distance_m = float(abs(v_dist))
                result.final_altitude_amsl = float(f_alt)
                result.final_relative_altitude = float(f_alt - home_coords['alt'])

            if result.cylinder_10m_success:
                result.terminal_reason = "success"

            logger.info(f"[{self.test_name_base}] Test {test_id} done. Criteria: "
                        + ", ".join(f"{c}={'✓' if achieved[c] else '✗'}" for c in self.all_criteria))

            self.controller.stop()
            if hasattr(self.controller, 'copy_flight_log'):
                self.controller.copy_flight_log(
                    position_index, trial_number,
                    success=bool(achieved.get('cylinder_10m')),
                    experiment=self.experiment,
                    platform="px4-jmavsim",
                )
            time.sleep(2)

        return result


def _run_all(test, attack_func, banner: str):
    logger.info(f"\n{'#'*80}\n# {banner}\n# {NUM_START_POSITIONS * NUM_TRIALS_PER_POSITION} flights, 4 criteria tracked per flight\n{'#'*80}\n")
    for position_idx in range(NUM_START_POSITIONS):
        for trial in range(1, NUM_TRIALS_PER_POSITION + 1):
            result = test.run_test_episode(position_idx, trial, attack_func)
            test.metrics.add_result(result.to_dict())
    log_file = test.metrics.save(print_summary=True)
    logger.info(f"Results saved to: {log_file}")


class BaselineCase1(BaselineTest):
    """Random bias, re-drawn every step."""

    def __init__(self, controller):
        super().__init__(controller, "baseline-random")

    def random_attack(self, step_count, position_index, current_state=None):
        xy = np.random.uniform(ATTACK_RANGE_MIN, ATTACK_RANGE_MAX, size=2).astype(np.float32)
        bias = np.array([xy[0], xy[1], 0.0], dtype=np.float32)
        if step_count % 10 == 0:
            logger.info(f"[{self.test_name_base}] step {step_count}: "
                        f"bias=[{bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}]")
        return bias

    def run_all_tests(self):
        _run_all(self, self.random_attack, "Baseline Case 1: Random Attack")


class BaselineCase2(BaselineTest):
    """Single directional bias, held constant within an episode."""

    def __init__(self, controller):
        super().__init__(controller, "baseline-directional")
        self.persistent_attack = None

    def directional_attack(self, step_count, position_index, current_state=None):
        if self.persistent_attack is None:
            self.persistent_attack = calculate_directional_attack(position_index, NUM_START_POSITIONS)
            logger.info(f"[{self.test_name_base}] Persistent bias for pos {position_index}: "
                        f"{self.persistent_attack}")
        return self.persistent_attack

    def run_test_episode(self, position_index, trial_number, attack_func):
        # Recompute per episode so trial 2 does not reuse trial 1's bias.
        self.persistent_attack = None
        return super().run_test_episode(position_index, trial_number, attack_func)

    def run_all_tests(self):
        _run_all(self, self.directional_attack, "Baseline Case 2: Directional Attack")


class BaselineCase3(BaselineTest):
    """Adaptive directional bias from live heading."""

    def __init__(self, controller):
        super().__init__(controller, "baseline-adaptive")

    def adaptive_directional_attack(self, step_count, position_index, current_state):
        current_lat, current_lon = current_state[0], current_state[1]
        quaternion_wxyz = current_state[9:13]
        _, _, yaw = Rotation.from_quat(quaternion_wxyz, scalar_first=True).as_euler('xyz')
        geodesic = Geodesic.WGS84.Inverse(current_lat, current_lon, TARGET_LAT, TARGET_LON)
        bearing = math.radians(geodesic['azi1'])
        relative_bearing = bearing - yaw
        magnitude = MAX_GYRO_BIAS
        x_bias = -magnitude * math.sin(relative_bearing)
        y_bias = magnitude * math.cos(relative_bearing)
        if step_count % 10 == 0:
            logger.info(f"[{self.test_name_base}] step {step_count}: "
                        f"yaw={math.degrees(yaw):.1f}°, bearing={geodesic['azi1']:.1f}°, "
                        f"bias=[{x_bias:.4f}, {y_bias:.4f}, 0]")
        return np.array([x_bias, y_bias, 0.0], dtype=np.float32)

    def run_all_tests(self):
        _run_all(self, self.adaptive_directional_attack, "Baseline Case 3: Adaptive Directional Attack")


def main():
    parser = argparse.ArgumentParser(description="Baseline Attack Tests")
    parser.add_argument("--case", type=int, required=True, choices=[1, 2, 3],
                       help="Test case: 1=Random Attack, 2=Directional Attack, 3=Adaptive Directional Attack")
    parser.add_argument("--speedup", type=int, default=SIMULATION_SPEEDUP,
                       help="Simulation speedup factor")
    args = parser.parse_args()

    controller = PX4JMAVSimController(instance_id=0, speed_factor=args.speedup)

    try:
        if args.case == 1:
            test = BaselineCase1(controller)
        elif args.case == 2:
            test = BaselineCase2(controller)
        else:
            test = BaselineCase3(controller)
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
