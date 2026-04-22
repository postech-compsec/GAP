"""Project-local logger creator for RLlib evaluation runs."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from ray.tune.logger import UnifiedLogger


def default_ray_results_dir() -> Path:
    project_root = Path(__file__).resolve().parents[3]
    return Path(
        os.environ.get(
            "GAP_RAY_LOG_DIR",
            project_root / "results" / "fresh" / "ray_results",
        )
    )


def make_logger_creator(run_name: str):
    base_dir = default_ray_results_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    def _creator(config):
        timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = tempfile.mkdtemp(prefix=f"{run_name}_{timestr}_", dir=str(base_dir))
        return UnifiedLogger(config, logdir, loggers=None)

    return _creator
