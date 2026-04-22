"""SITL auto-eval of the sim-to-real model in PX4 + jMAVSim.

Mirrors primary/evaluate_multi_criteria.py: 12 parallel workers, one PX4 SITL
per worker, 12 equally-spaced start points on a spawn circle. The only
differences are (1) the env uses the onboard-sensor observation path
(matching what the model sees on the physical drone) and (2) the observation
is 25-dim (adds body-frame accel and 2D horizontal distance).

Run via `experiments/run_rq6.sh --mode approx` (wrapper sets
GAP_RESULTS_DIR) or invoke the script directly — in the latter case this
module defaults GAP_RESULTS_DIR to `results/fresh/rq6/sim_evaluation/` so
per-worker JSONs always land under rq6/ instead of the cwd.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

# Bootstrap PYTHONPATH so Ray workers can import `gap.*` and `evaluation.*`.
_SRC_DIR = str(Path(__file__).resolve().parents[3])   # .../GAP/src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_existing = os.environ.get("PYTHONPATH", "")
if _SRC_DIR not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = _SRC_DIR + (os.pathsep + _existing if _existing else "")

# Default output dir so direct invocation (without run_rq6.sh) still writes
# JSONs under the expected RQ6 tree instead of the cwd. Ray workers inherit
# the environ block set here because MetricsCollector reads GAP_RESULTS_DIR
# at env __init__ time in each worker subprocess.
if not os.environ.get("GAP_RESULTS_DIR"):
    _project_root = Path(__file__).resolve().parents[4]   # .../GAP
    _default_out = _project_root / "results" / "fresh" / "rq6" / "sim_evaluation"
    _default_out.mkdir(parents=True, exist_ok=True)
    os.environ["GAP_RESULTS_DIR"] = str(_default_out)

# Local imports (run with cwd=src/evaluation/sim_to_real/sim/).
from evaluation.common.ray_logging import make_logger_creator
from gym_env import PX4RLEnvSimToRealEval
from config import (
    GPU_WORKER,
    NUM_EVAL_WORKERS, RANDOM_START_RADIUS_M,
    TARGET_LAT, TARGET_LON, METERS_PER_DEGREE_LAT,
)
from gap.asymmetric_rl_module import AsymmetricLSTMModule


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _resolve_checkpoint_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    return str(path)


def get_start_points(num_points: int, radius: float):
    points = []
    for i in range(num_points):
        theta = 2 * math.pi * i / max(1, num_points)
        dn, de = radius * math.cos(theta), radius * math.sin(theta)
        dlat = dn / METERS_PER_DEGREE_LAT
        dlon = de / (METERS_PER_DEGREE_LAT * math.cos(math.radians(TARGET_LAT)))
        points.append({"lat": TARGET_LAT + dlat, "lon": TARGET_LON + dlon})
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to sim-to-real model checkpoint")
    parser.add_argument("--num_trials", type=int, default=2,
                        help="Trials per worker (2 for approx, 100 for full)")
    args = parser.parse_args()

    start_points = get_start_points(NUM_EVAL_WORKERS, RANDOM_START_RADIUS_M)
    ray.init(ignore_reinit_error=True)

    print("=== Sim-to-Real Model SITL Auto-Eval (PX4-jMAVSim, onboard-sensor obs) ===")
    register_env("sim-to-real-eval", lambda cfg: PX4RLEnvSimToRealEval(cfg))

    total_episodes = NUM_EVAL_WORKERS * args.num_trials
    config = (
        APPOConfig()
        .environment("sim-to-real-eval", env_config={})
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(module_class=AsymmetricLSTMModule),
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=NUM_EVAL_WORKERS,
            evaluation_duration=total_episodes,
            evaluation_duration_unit="episodes",
            evaluation_force_reset_envs_before_iteration=True,
            evaluation_sample_timeout_s=36000,
            evaluation_parallel_to_training=False,
            evaluation_config={
                "explore": False,
                # Keep the paper's 12 parallel workers even on small VMs.
                "num_cpus_per_env_runner": 0,
                "num_gpus_per_env_runner": (
                    (1.0 / float(max(1, NUM_EVAL_WORKERS)))
                    if _has_gpu() else 0
                ),
                "env_config": {"start_points": start_points},
            },
        )
    )

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    algo = config.build_algo(logger_creator=make_logger_creator("rq6_sim"))
    algo.restore(checkpoint_path)

    print(f"Running {total_episodes} episodes across {NUM_EVAL_WORKERS} workers")
    results = algo.evaluate()
    print("=== Done ===")
    print(results)
    print("\nResults saved per worker: "
          "<ts>_gap-sim-to-real_px4-jmavsim_w<NN>.json")


if __name__ == "__main__":
    main()
