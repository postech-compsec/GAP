import argparse
import os
import sys
from pathlib import Path

import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

_SRC_DIR = str(Path(__file__).resolve().parents[3])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
_existing = os.environ.get("PYTHONPATH", "")
if _SRC_DIR not in _existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = _SRC_DIR + (os.pathsep + _existing if _existing else "")

from evaluation.sim_to_real.real.gym_env import PX4RLEnv
from evaluation.common.ray_logging import make_logger_creator
from gap.asymmetric_rl_module import AsymmetricLSTMModule


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
    args = parser.parse_args()

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        num_gpus=0,
    )

    print("=== Asymmetric Evaluation (actor-only inference) ===")
    register_env("eval-env", lambda cfg: PX4RLEnv(cfg))

    config = (
        APPOConfig()
        .environment("eval-env", env_config={})
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
            evaluation_interval=1,
            evaluation_num_workers=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_force_reset_envs_before_iteration=True,
            evaluation_sample_timeout_s=600,
            evaluation_parallel_to_training=False,
            evaluation_config={
                "explore": False,
                "num_cpus_per_env_runner": 4,
                "num_gpus_per_env_runner": 0,
            },
        )
    )

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    algo = config.build_algo(logger_creator=make_logger_creator("rq6_real"))
    algo.restore(checkpoint_path)

    print("Starting evaluation: target 1 episode across 1 worker")
    results = algo.evaluate()
    print("=== Evaluation Results ===")
    print(results)


if __name__ == "__main__":
    main()
