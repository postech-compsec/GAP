"""Evaluate a GAP checkpoint against all 4 success criteria in one run
(sphere/cylinder × 10/20 m). Actor-only inference, 12 parallel Ray workers.
"""

import argparse, ray, math, numpy as np
import gymnasium as gym
import os
from pathlib import Path
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from gym_env_eval import PX4RLEnvEval
from evaluation.common.ray_logging import make_logger_creator
from gap.asymmetric_rl_module import AsymmetricLSTMModule
from config import RANDOM_START_RADIUS_M, TARGET_LAT, TARGET_LON, METERS_PER_DEGREE_LAT, NUM_EVAL_WORKERS


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def get_start_points(num_points: int, radius: float):
    points = []
    for i in range(num_points):
        theta = 2 * math.pi * i / max(1, num_points)
        dn, de = radius * math.cos(theta), radius * math.sin(theta)
        dlat = dn / METERS_PER_DEGREE_LAT
        dlon = de / (METERS_PER_DEGREE_LAT * math.cos(math.radians(TARGET_LAT)))
        points.append({"lat": TARGET_LAT + dlat, "lon": TARGET_LON + dlon})
    return points


def make_env_creator(base_env_cls, start_points):
    """Shim worker_index 0→1-based (PX4RLEnvEval indexes spawn points by worker
    number - 1) and inject the shared start_points list."""

    def _creator(env_config):
        try:
            raw_worker = int(getattr(env_config, "worker_index", 0))
        except Exception:
            raw_worker = int(env_config.get("worker_index", 0)) if isinstance(env_config, dict) else 0
        cfg = dict(env_config) if isinstance(env_config, dict) else {}
        cfg["worker_index"] = raw_worker + 1
        cfg["start_points"] = start_points
        return base_env_cls(cfg)

    return _creator


def _primary_variant() -> str | None:
    tag = os.environ.get("GAP_PRIMARY_VARIANT", "").strip().lower()
    return tag or None


def _run_name() -> str:
    explicit = os.environ.get("GAP_RAY_RUN_NAME", "").strip()
    if explicit:
        return explicit

    variant = _primary_variant()
    if variant:
        return f"rq2_gap_px4_jmavsim_{variant}"
    return "primary_gap_px4_jmavsim"


def _resolve_checkpoint_path(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, help="Path to checkpoint directory or file")
    parser.add_argument("--num_trials", type=int, default=100, help="Target trials per worker (episodes per spawn point)")
    args = parser.parse_args()

    start_points = get_start_points(NUM_EVAL_WORKERS, RANDOM_START_RADIUS_M)

    ray.init(ignore_reinit_error=True)

    print("=== Multi-Criteria Evaluation (all 4 success criteria simultaneously) ===")
    register_env("eval-env-multi", lambda cfg: PX4RLEnvEval(cfg))

    total_episodes = NUM_EVAL_WORKERS * args.num_trials
    config = (
        APPOConfig()
        .environment("eval-env-multi", env_config={})
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
            evaluation_sample_timeout_s=36000,  # 10 h — RL evals can be long
            evaluation_parallel_to_training=False,
            evaluation_config={
                "explore": False,
                # Keep the paper's 12 parallel workers even on small VMs.
                "num_cpus_per_env_runner": 0,
                # Fractional-share the single GPU across all workers; fall
                # back to 0 on CPU-only hosts so Ray can still schedule.
                "num_gpus_per_env_runner": (
                    (1.0 / float(max(1, NUM_EVAL_WORKERS)))
                    if _has_gpu() else 0
                ),
                "env_config": {
                    "start_points": start_points,
                },
            },
        )
    )

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    algo = config.build_algo(logger_creator=make_logger_creator(_run_name()))
    algo.restore(checkpoint_path)

    print(f"Running {total_episodes} episodes across {NUM_EVAL_WORKERS} workers")
    print()

    results = algo.evaluate()

    print("=== Evaluation Results ===")
    print(results)

    print("\nResults are saved as one JSON per worker:")
    print("  <ts>_gap_px4-jmavsim[_<variant>]_w<NN>.json")


if __name__ == "__main__":
    main()
