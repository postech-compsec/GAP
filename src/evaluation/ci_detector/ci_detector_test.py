#!/usr/bin/env python3
"""Run one CI-detector trial for one target direction."""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from geographiclib.geodesic import Geodesic
from scipy.spatial.transform import Rotation
import torch
import ray
import gymnasium as gym

import os as _os
_SRC_DIR = str(Path(__file__).resolve().parents[2])   # .../GAP/src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_existing = _os.environ.get("PYTHONPATH", "")
if _SRC_DIR not in _existing.split(_os.pathsep):
    _os.environ["PYTHONPATH"] = _SRC_DIR + (_os.pathsep + _existing if _existing else "")

from evaluation.common.controllers.ardupilot_legacy_controller import ArdupilotLegacyController
from evaluation.common.ray_logging import make_logger_creator
from evaluation.common.metrics import MetricsCollector, NumpyEncoder
from gap.asymmetric_rl_module import AsymmetricLSTMModule
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.columns import Columns
from ray.tune.registry import register_env

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

DISTANCE_M = 220
MAX_BIAS = 0.06
TIMEOUT = 300
TAKEOFF_ALT = 60.0

DIRECTION_NAMES = [
    "North", "NNE", "ENE", "East", "ESE", "SSE",
    "South", "SSW", "WSW", "West", "WNW", "NNW"
]

def calculate_target_position(origin_lat, origin_lon, direction_index):
    """Return the target point for one of the 12 directions."""
    bearing_deg = (direction_index - 1) * 30.0

    geod = Geodesic.WGS84
    result = geod.Direct(origin_lat, origin_lon, bearing_deg, DISTANCE_M)

    target_lat = result['lat2']
    target_lon = result['lon2']
    direction_name = DIRECTION_NAMES[direction_index - 1]

    return target_lat, target_lon, bearing_deg, direction_name

def check_distance(current_lat, current_lon, current_alt, target_lat, target_lon):
    """Return horizontal and 3D distance to target."""
    geod = Geodesic.WGS84
    g = geod.Inverse(current_lat, current_lon, target_lat, target_lon)
    h_dist = g['s12']
    alt_diff = current_alt - 60.0
    dist_3d = np.sqrt(h_dist**2 + alt_diff**2)
    return h_dist, dist_3d

class ObservationProcessor:
    """Map raw state into the actor observation."""

    def __init__(self, initial_hil_state, episode_start_time, target_lat, target_lon):
        self.initial_hil = initial_hil_state
        self.episode_start_time = episode_start_time
        self.target_lat = target_lat
        self.target_lon = target_lon

    def process_observation(self, hil_state):
        """Return the 21-D actor observation."""
        latitude, longitude, altitude_amsl = hil_state[0:3]
        vx, vy, vz = hil_state[3:6]
        ax, ay, az = hil_state[6:9]
        quaternion_wxyz = hil_state[9:13]
        rollspeed, pitchspeed, yawspeed = hil_state[13:16]

        time_since_reset = time.monotonic() - self.episode_start_time
        normalized_time = time_since_reset / TIMEOUT

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

        SUCCESS_RADIUS = 10.0
        is_in_sphere = 1 if distance_to_target <= SUCCESS_RADIUS else 0

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

def load_rl_model(checkpoint_path):
    """Load the trained RL policy for inference."""
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    logger.info(f"Loading RL model from: {checkpoint_path}")

    cross_platform_path = str(Path(__file__).parent.parent / "cross_platform")
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        num_gpus=0,
        runtime_env={
            "py_modules": [cross_platform_path],
            "env_vars": {"PYTHONPATH": cross_platform_path}
        }
    )

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

    register_env("dummy-ci-detector-env", lambda cfg: DummyEnv(cfg))

    config = (
        APPOConfig()
        .environment("dummy-ci-detector-env", env_config={})
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

    algo = config.build_algo(logger_creator=make_logger_creator("rq4_ci_detector"))
    algo.restore(checkpoint_path)

    logger.info("RL model loaded successfully")
    return algo

def predict_action(rl_model, observation, prev_action, lstm_state=None):
    """Run one deterministic policy step."""
    if rl_model is None:
        logger.error("RL model not loaded")
        return np.zeros(2, dtype=np.float32), lstm_state

    actor_obs_with_action = np.concatenate([observation, prev_action], dtype=np.float32)
    rl_module = rl_model.get_module()
    if lstm_state is None:
        lstm_state = rl_module.get_initial_state()

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
        logger.error("No action_dist_inputs in model output")
        action = np.zeros(2, dtype=np.float32)

    new_state = output.get(Columns.STATE_OUT, {})
    new_state_dict = {}
    for key, value in new_state.items():
        if torch.is_tensor(value):
            new_state_dict[key] = value.cpu().numpy()
        else:
            new_state_dict[key] = value

    return action, new_state_dict if new_state_dict else lstm_state

def main():
    import os
    parser = argparse.ArgumentParser(description='CI-Detector Evasion Test (single trial)')
    parser.add_argument('--loc', type=int, required=True, choices=range(1, 13),
                        help='Target direction (1..12, clockwise from North)')
    parser.add_argument('--trial', type=int, default=1,
                        help='Trial number within this direction (1 or 2)')
    parser.add_argument('--checkpoint', type=str,
                        default=os.environ.get('GAP_MODEL_PATH',
                            str(Path(__file__).resolve().parents[3] / "src/gap/models/gap_model")),
                        help='RL model checkpoint path (default: bundled gap_model)')
    parser.add_argument('--connection', type=str, default="udpin:0.0.0.0:17000",
                        help='MAVLink connection string')
    args = parser.parse_args()

    log_dir = os.environ.get("GAP_RESULTS_DIR",
        str(Path(__file__).resolve().parents[3] / "results/fresh/rq4"))
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    metrics = MetricsCollector(
        output_dir=log_dir,
        experiment="gap-ci-detector",
        platform="ardupilot-vm",
        worker=args.loc,       # 12-point direction circle maps to the worker slot
        variant=f"trial{args.trial}",
    )

    controller = None

    try:
        logger.info("="*80)
        logger.info("CI-Detector Directional Attack Test")
        logger.info("="*80)

        logger.info(f"Connecting to {args.connection}...")
        controller = ArdupilotLegacyController(
            frame_type="quad",
            connection_string=args.connection,
            speedup=1
        )

        logger.info("Getting origin position...")
        state = controller.get_state(timeout=5.0)
        if state is None:
            logger.error("Failed to get state. Is vehicle airborne at 60m?")
            return 1

        origin_lat, origin_lon, origin_alt = state[0], state[1], state[2]
        logger.info(f"Origin: lat={origin_lat:.6f}, lon={origin_lon:.6f}, alt={origin_alt:.1f}m")

        target_lat, target_lon, bearing, direction_name = calculate_target_position(
            origin_lat, origin_lon, args.loc
        )

        logger.info(f"\nTarget Location #{args.loc}: {direction_name} ({bearing:.0f}°)")
        logger.info(f"Target: lat={target_lat:.6f}, lon={target_lon:.6f}, alt=60m")
        logger.info(f"Distance: {DISTANCE_M}m")

        h_dist, dist_3d = check_distance(origin_lat, origin_lon, origin_alt, target_lat, target_lon)
        logger.info(f"Current distance: horizontal={h_dist:.1f}m, 3D={dist_3d:.1f}m")

        # Enable CI-detector
        #logger.info("\nEnabling CI-detector...")
        #controller.enable_ci_detector()
        #time.sleep(1)

        rl_model = load_rl_model(args.checkpoint)

        rl_module = rl_model.get_module()
        lstm_state = rl_module.get_initial_state()
        prev_action = np.zeros(3, dtype=np.float32)

        # CI-detector eval starts already hovering at TAKEOFF_ALT.
        episode_start_time = time.monotonic()
        initial_hil_adjusted = state.copy()
        initial_hil_adjusted[2] = state[2] - TAKEOFF_ALT

        obs_processor = ObservationProcessor(
            initial_hil_state=initial_hil_adjusted,
            episode_start_time=episode_start_time,
            target_lat=target_lat,
            target_lon=target_lon
        )

        observation = obs_processor.process_observation(state)

        logger.info("\nStarting RL attack...")
        logger.info("="*80)
        start_time = time.time()
        step = 0

        success_cyl20 = False
        success_sph20 = False
        success_cyl10 = False
        success_sph10 = False

        time_cyl20 = None
        time_sph20 = None
        time_cyl10 = None
        time_sph10 = None

        distance_cyl20 = None
        distance_sph20 = None
        distance_cyl10 = None
        distance_sph10 = None

        altitude_cyl20 = None
        altitude_sph20 = None
        altitude_cyl10 = None
        altitude_sph10 = None

        step_cyl20 = None
        step_sph20 = None
        step_cyl10 = None
        step_sph10 = None

        detected_cyl20 = None
        detected_sph20 = None
        detected_cyl10 = None
        detected_sph10 = None

        controller.drain_all()

        while time.time() - start_time < TIMEOUT:
            step_start_time = time.monotonic()

            action, lstm_state = predict_action(rl_model, observation, prev_action, lstm_state)
            bias = np.array([action[0] * MAX_BIAS, action[1] * MAX_BIAS, 0.0])

            controller.set_gyro_bias(bias)
            step += 1
            prev_action = bias

            # Keep checking success during each 1 s injection window.
            step_duration = 1.0

            controller.drain_all()
            last_state = None
            while time.monotonic() - step_start_time < step_duration:
                state = controller.get_state(timeout=0.5)
                if state is None:
                    continue

                last_state = state

                lat, lon, alt = state[0], state[1], state[2]
                h_dist, dist_3d = check_distance(lat, lon, alt, target_lat, target_lon)
                elapsed = time.time() - start_time

                controller.check_attack_detection(episode_start_time=start_time)
                detection_info = controller.get_detection_info()
                is_detected = detection_info['detected']

                if h_dist <= 20.0 and not success_cyl20:
                    success_cyl20 = True
                    time_cyl20 = elapsed
                    distance_cyl20 = dist_3d
                    altitude_cyl20 = alt
                    step_cyl20 = step
                    detected_cyl20 = is_detected
                    detect_status = "DETECTED" if is_detected else "UNDETECTED"
                    logger.info(f"✓ Cylinder 20m SUCCESS at t={elapsed:.1f}s | dist={dist_3d:.1f}m | step={step} | CI: {detect_status}")

                if dist_3d <= 20.0 and not success_sph20:
                    success_sph20 = True
                    time_sph20 = elapsed
                    distance_sph20 = dist_3d
                    altitude_sph20 = alt
                    step_sph20 = step
                    detected_sph20 = is_detected
                    detect_status = "DETECTED" if is_detected else "UNDETECTED"
                    logger.info(f"✓ Sphere 20m SUCCESS at t={elapsed:.1f}s | dist={dist_3d:.1f}m | step={step} | CI: {detect_status}")

                if h_dist <= 10.0 and not success_cyl10:
                    success_cyl10 = True
                    time_cyl10 = elapsed
                    distance_cyl10 = dist_3d
                    altitude_cyl10 = alt
                    step_cyl10 = step
                    detected_cyl10 = is_detected
                    detect_status = "DETECTED" if is_detected else "UNDETECTED"
                    logger.info(f"✓ Cylinder 10m SUCCESS at t={elapsed:.1f}s | dist={dist_3d:.1f}m | step={step} | CI: {detect_status}")

                if dist_3d <= 10.0 and not success_sph10:
                    success_sph10 = True
                    time_sph10 = elapsed
                    distance_sph10 = dist_3d
                    altitude_sph10 = alt
                    step_sph10 = step
                    detected_sph10 = is_detected
                    detect_status = "DETECTED" if is_detected else "UNDETECTED"
                    logger.info(f"✓ Sphere 10m SUCCESS at t={elapsed:.1f}s | dist={dist_3d:.1f}m | step={step} | CI: {detect_status}")

                if success_cyl20 and success_sph20 and success_cyl10 and success_sph10:
                    logger.info(f"\n{'='*80}")
                    logger.info("All success conditions met! Early termination.")
                    logger.info(f"{'='*80}")
                    break

            if last_state is not None:
                observation = obs_processor.process_observation(last_state)
                state = last_state
            else:
                logger.warning("State timeout - no state received in inner loop")
                break

            if success_cyl20 and success_sph20 and success_cyl10 and success_sph10:
                break

            controller.check_attack_detection(episode_start_time=start_time)

            if step % 5 == 0:
                detection_info = controller.get_detection_info()
                lat, lon, alt = state[0], state[1], state[2]
                h_dist, dist_3d = check_distance(lat, lon, alt, target_lat, target_lon)
                elapsed = time.time() - start_time
                logger.info(
                    f"t={elapsed:.1f}s | h_dist={h_dist:.1f}m | 3d_dist={dist_3d:.1f}m | "
                    f"alt={alt:.1f}m | detected={detection_info['detected']}"
                )

        detection_info = controller.get_detection_info()
        elapsed_total = time.time() - start_time

        logger.info("\n" + "="*80)
        logger.info("Test Results:")
        logger.info("="*80)
        logger.info(f"Location: #{args.loc} - {direction_name} ({bearing:.0f}°)")

        time_str_cyl20 = f"({time_cyl20:.1f}s)" if time_cyl20 else ""
        detect_str_cyl20 = ""
        if success_cyl20:
            detect_str_cyl20 = f" | CI: {'DETECTED' if detected_cyl20 else 'UNDETECTED'}"
        logger.info(f"Cylinder 20m: {'✓ SUCCESS' if success_cyl20 else '✗ FAILED'} {time_str_cyl20}{detect_str_cyl20}")

        time_str_sph20 = f"({time_sph20:.1f}s)" if time_sph20 else ""
        detect_str_sph20 = ""
        if success_sph20:
            detect_str_sph20 = f" | CI: {'DETECTED' if detected_sph20 else 'UNDETECTED'}"
        logger.info(f"Sphere 20m:   {'✓ SUCCESS' if success_sph20 else '✗ FAILED'} {time_str_sph20}{detect_str_sph20}")

        time_str_cyl10 = f"({time_cyl10:.1f}s)" if time_cyl10 else ""
        detect_str_cyl10 = ""
        if success_cyl10:
            detect_str_cyl10 = f" | CI: {'DETECTED' if detected_cyl10 else 'UNDETECTED'}"
        logger.info(f"Cylinder 10m: {'✓ SUCCESS' if success_cyl10 else '✗ FAILED'} {time_str_cyl10}{detect_str_cyl10}")

        time_str_sph10 = f"({time_sph10:.1f}s)" if time_sph10 else ""
        detect_str_sph10 = ""
        if success_sph10:
            detect_str_sph10 = f" | CI: {'DETECTED' if detected_sph10 else 'UNDETECTED'}"
        logger.info(f"Sphere 10m:   {'✓ SUCCESS' if success_sph10 else '✗ FAILED'} {time_str_sph10}{detect_str_sph10}")

        detected_time_str = f"({detection_info['detection_time']:.1f}s)" if detection_info['detected'] else ""
        logger.info(f"Overall CI-Detector:  {'DETECTED' if detection_info['detected'] else 'UNDETECTED'} {detected_time_str}")
        logger.info(f"Total Time: {elapsed_total:.1f}s")
        logger.info("="*80)

        final_state = controller.get_state(timeout=1.0)
        if final_state is not None:
            final_lat, final_lon, final_alt = final_state[0], final_state[1], final_state[2]
            final_h_dist, final_dist_3d = check_distance(final_lat, final_lon, final_alt, target_lat, target_lon)
        else:
            final_h_dist = float('inf')
            final_dist_3d = float('inf')
            final_alt = None

        result = {
            "location": args.loc,
            "direction": direction_name,
            "bearing_deg": bearing,
            "cylinder_20m": success_cyl20,
            "sphere_20m": success_sph20,
            "cylinder_10m": success_cyl10,
            "sphere_10m": success_sph10,
            "detected_cyl20": detected_cyl20,
            "detected_sph20": detected_sph20,
            "detected_cyl10": detected_cyl10,
            "detected_sph10": detected_sph10,
            "ci_detected_overall": detection_info['detected'],
            "time_cyl20": time_cyl20,
            "time_sph20": time_sph20,
            "time_cyl10": time_cyl10,
            "time_sph10": time_sph10,
            "distance_cyl20": distance_cyl20,
            "distance_sph20": distance_sph20,
            "distance_cyl10": distance_cyl10,
            "distance_sph10": distance_sph10,
            "altitude_cyl20": altitude_cyl20,
            "altitude_sph20": altitude_sph20,
            "altitude_cyl10": altitude_cyl10,
            "altitude_sph10": altitude_sph10,
            "step_cyl20": step_cyl20,
            "step_sph20": step_sph20,
            "step_cyl10": step_cyl10,
            "step_sph10": step_sph10,
            "time_detected": detection_info['detection_time'] if detection_info['detected'] else None,
            "total_time": elapsed_total,
            "time_spent_s": elapsed_total,
            "final_distance_m": final_dist_3d,
            "horizontal_distance_m": final_h_dist,
            "vertical_distance_m": abs(final_alt - 60.0) if final_alt else None,
            "final_altitude_amsl": final_alt,
            "attack_steps": step,
            "success": success_cyl20 or success_sph20 or success_cyl10 or success_sph10,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "target_alt": 60.0
        }

        log_file = metrics.log_file
        with open(log_file, 'w') as f:
            json.dump(result, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {log_file}")
        logger.info(f"{'='*80}\n")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nKeyboard interrupt received. Saving results...")

        elapsed_time = time.time() - start_time if 'start_time' in locals() else 0

        cyl20 = success_cyl20 if 'success_cyl20' in locals() else False
        sph20 = success_sph20 if 'success_sph20' in locals() else False
        cyl10 = success_cyl10 if 'success_cyl10' in locals() else False
        sph10 = success_sph10 if 'success_sph10' in locals() else False
        overall_success = cyl20 and sph20 and cyl10 and sph10

        current_state = None
        final_h_dist = None
        final_dist_3d = None
        if 'controller' in locals() and 'target_lat' in locals() and 'target_lon' in locals():
            try:
                current_state = controller.get_state(timeout=1.0)
                if current_state is not None:
                    lat, lon, alt = current_state[0], current_state[1], current_state[2]
                    final_h_dist, final_dist_3d = check_distance(lat, lon, alt, target_lat, target_lon)
            except:
                pass

        det_cyl20 = detected_cyl20 if 'detected_cyl20' in locals() else None
        det_sph20 = detected_sph20 if 'detected_sph20' in locals() else None
        det_cyl10 = detected_cyl10 if 'detected_cyl10' in locals() else None
        det_sph10 = detected_sph10 if 'detected_sph10' in locals() else None

        t_cyl20 = time_cyl20 if 'time_cyl20' in locals() else None
        t_sph20 = time_sph20 if 'time_sph20' in locals() else None
        t_cyl10 = time_cyl10 if 'time_cyl10' in locals() else None
        t_sph10 = time_sph10 if 'time_sph10' in locals() else None

        dist_cyl20 = distance_cyl20 if 'distance_cyl20' in locals() else None
        dist_sph20 = distance_sph20 if 'distance_sph20' in locals() else None
        dist_cyl10 = distance_cyl10 if 'distance_cyl10' in locals() else None
        dist_sph10 = distance_sph10 if 'distance_sph10' in locals() else None

        alt_cyl20 = altitude_cyl20 if 'altitude_cyl20' in locals() else None
        alt_sph20 = altitude_sph20 if 'altitude_sph20' in locals() else None
        alt_cyl10 = altitude_cyl10 if 'altitude_cyl10' in locals() else None
        alt_sph10 = altitude_sph10 if 'altitude_sph10' in locals() else None

        s_cyl20 = step_cyl20 if 'step_cyl20' in locals() else None
        s_sph20 = step_sph20 if 'step_sph20' in locals() else None
        s_cyl10 = step_cyl10 if 'step_cyl10' in locals() else None
        s_sph10 = step_sph10 if 'step_sph10' in locals() else None

        detected_values = [v for v in [det_cyl20, det_sph20, det_cyl10, det_sph10] if v is not None]
        ci_detected_overall = any(detected_values) if detected_values else None

        partial_result = {
            "location": args.loc,
            "interrupted": True,
            "success": overall_success,

            "cylinder_20m": cyl20,
            "sphere_20m": sph20,
            "cylinder_10m": cyl10,
            "sphere_10m": sph10,
            "detected_cyl20": det_cyl20,
            "detected_sph20": det_sph20,
            "detected_cyl10": det_cyl10,
            "detected_sph10": det_sph10,
            "ci_detected_overall": ci_detected_overall,
            "time_cyl20": t_cyl20,
            "time_sph20": t_sph20,
            "time_cyl10": t_cyl10,
            "time_sph10": t_sph10,
            "distance_cyl20": dist_cyl20,
            "distance_sph20": dist_sph20,
            "distance_cyl10": dist_cyl10,
            "distance_sph10": dist_sph10,
            "altitude_cyl20": alt_cyl20,
            "altitude_sph20": alt_sph20,
            "altitude_cyl10": alt_cyl10,
            "altitude_sph10": alt_sph10,
            "step_cyl20": s_cyl20,
            "step_sph20": s_sph20,
            "step_cyl10": s_cyl10,
            "step_sph10": s_sph10,
            "total_time": elapsed_time,
            "time_spent_s": elapsed_time,
            "final_distance_m": final_dist_3d if final_dist_3d is not None else float('inf'),
            "attack_steps": step if 'step' in locals() else 0,
        }

        log_file = metrics.log_file
        with open(log_file, 'w') as f:
            json.dump(partial_result, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {log_file}")
        logger.info(f"{'='*80}\n")
        return 130

    finally:
        if controller is not None:
            controller.close()

if __name__ == "__main__":
    sys.exit(main())
