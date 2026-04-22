"""Unified JSON schema for all evaluation pipelines (primary RL, baseline,
cross-platform, CI-detector, sim-to-real).

All files follow the same timestamp-first filename convention so `ls` sorts
chronologically; each file carries a `summary` block + per-episode `results`
tracking all 4 success criteria at once.
"""

import json
import os
import shutil
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


CRITERIA = ["sphere_20m", "cylinder_20m", "sphere_10m", "cylinder_10m"]


def raw_log_mode() -> str:
    mode = os.environ.get("GAP_RAW_LOG_MODE", "off").strip().lower()
    return mode if mode in {"move", "off"} else "off"


def export_raw_log(src_path: str, dest_path: str) -> str:
    """Handle one raw flight log according to GAP_RAW_LOG_MODE.

    Modes:
    - move: move into results to avoid duplication
    - off: delete the source file and do not keep a raw export
    """
    mode = raw_log_mode()
    if mode == "off":
        os.remove(src_path)
        return "deleted"

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.move(src_path, dest_path)
    return "moved"


def make_output_path(
    output_dir: str,
    experiment: str,
    platform: str,
    frame: Optional[str] = None,
    variant: Optional[str] = None,
    worker: Optional[int] = None,
    episode: Optional[int] = None,
    outcome: Optional[str] = None,
    timestamp: Optional[str] = None,
    ext: str = "json",
) -> str:
    """Return ``<ts>_<experiment>_<platform>[_<frame>][_<variant>][_w<NN>][_ep<NNN>][_<outcome>].<ext>``.

    Shared by `MetricsCollector` (json), `PX4Controller.organize_ulog_files`
    (ulg), and `ArdupilotController.copy_flight_log` (BIN). Optional slots
    are dropped when None.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [ts, experiment, platform]
    if frame:
        parts.append(frame)
    if variant:
        parts.append(variant)
    if worker is not None:
        parts.append(f"w{worker:02d}")
    if episode is not None:
        parts.append(f"ep{episode:03d}")
    if outcome:
        parts.append(outcome)
    return os.path.join(output_dir, "_".join(parts) + "." + ext)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class TestResult:
    """One episode, tracking all 4 success criteria simultaneously."""

    def __init__(self, test_id: int, position_index: int, trial_number: int):
        self.test_id = test_id
        self.position_index = position_index
        self.trial_number = trial_number

        self.sphere_20m_success = False
        self.sphere_20m_time_s = None
        self.sphere_20m_distance_m = None

        self.cylinder_20m_success = False
        self.cylinder_20m_time_s = None
        self.cylinder_20m_horizontal_m = None

        self.sphere_10m_success = False
        self.sphere_10m_time_s = None
        self.sphere_10m_distance_m = None

        self.cylinder_10m_success = False
        self.cylinder_10m_time_s = None
        self.cylinder_10m_horizontal_m = None

        self.final_distance_m = None
        self.horizontal_distance_m = None
        self.vertical_distance_m = None
        self.final_altitude_amsl = None
        self.final_relative_altitude = None
        self.time_spent_s = None
        self.attack_steps = 0
        self.terminal_reason = None
        self.failsafe_occurred = False
        self.failsafe_events: List = []
        self.additional_data: Dict[str, Any] = {}

    def record_criterion(self, criterion: str, time_s: float, distance_m: float,
                         horizontal_m: float) -> None:
        setattr(self, f"{criterion}_success", True)
        setattr(self, f"{criterion}_time_s", time_s)
        # Sphere tracks 3D; cylinder tracks horizontal-only.
        if criterion.startswith("sphere"):
            setattr(self, f"{criterion}_distance_m", distance_m)
        else:
            setattr(self, f"{criterion}_horizontal_m", horizontal_m)

    def to_dict(self) -> Dict[str, Any]:
        def _r2(v): return round(v, 2) if v is not None else None
        def _r3(v): return round(v, 3) if v is not None else None

        return {
            "test_id": self.test_id,
            "position_index": self.position_index,
            "clock_position": self.position_index if self.position_index > 0 else 12,
            "trial_number": self.trial_number,

            "sphere_20m_success": self.sphere_20m_success,
            "sphere_20m_time_s": _r3(self.sphere_20m_time_s),
            "sphere_20m_distance_m": _r2(self.sphere_20m_distance_m),

            "cylinder_20m_success": self.cylinder_20m_success,
            "cylinder_20m_time_s": _r3(self.cylinder_20m_time_s),
            "cylinder_20m_horizontal_m": _r2(self.cylinder_20m_horizontal_m),

            "sphere_10m_success": self.sphere_10m_success,
            "sphere_10m_time_s": _r3(self.sphere_10m_time_s),
            "sphere_10m_distance_m": _r2(self.sphere_10m_distance_m),

            "cylinder_10m_success": self.cylinder_10m_success,
            "cylinder_10m_time_s": _r3(self.cylinder_10m_time_s),
            "cylinder_10m_horizontal_m": _r2(self.cylinder_10m_horizontal_m),

            "final_distance_m": _r2(self.final_distance_m),
            "horizontal_distance_m": _r2(self.horizontal_distance_m),
            "vertical_distance_m": _r2(self.vertical_distance_m),
            "final_altitude_amsl": _r2(self.final_altitude_amsl),
            "final_relative_altitude": _r2(self.final_relative_altitude),
            "time_spent_s": _r3(self.time_spent_s),
            "attack_steps": self.attack_steps,
            "terminal_reason": self.terminal_reason,
            "failsafe_occurred": self.failsafe_occurred,
            "failsafe_events": self.failsafe_events,
            **self.additional_data,
        }

class MetricsCollector:
    """Aggregates per-episode results into one JSON via the shared filename
    convention (see `make_output_path`)."""

    def __init__(
        self,
        output_dir: str,
        experiment: str,
        platform: str,
        frame: Optional[str] = None,
        variant: Optional[str] = None,
        worker: Optional[int] = None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = make_output_path(
            output_dir, experiment, platform,
            frame=frame, variant=variant, worker=worker,
            timestamp=self.timestamp, ext="json",
        )
        self.test_name = os.path.basename(self.log_file)[:-len(".json")]
        self.results: List[Dict[str, Any]] = []
        self.start_time = time.monotonic()

    def add_result(self, result: Dict[str, Any]) -> None:
        self.results.append(result)

    @staticmethod
    def _stats(values):
        if not values:
            return {"mean": None, "std": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": round(float(np.mean(arr)), 2),
            "std":  round(float(np.std(arr)), 2),
            "min":  round(float(np.min(arr)), 2),
            "max":  round(float(np.max(arr)), 2),
        }

    def get_summary(self) -> Dict[str, Any]:
        n = len(self.results)
        summary: Dict[str, Any] = {
            "test_name": self.test_name,
            "timestamp": self.timestamp,
            "total_tests": n,
        }
        if n == 0:
            return summary

        for crit in CRITERIA:
            successes = sum(1 for r in self.results if r.get(f"{crit}_success"))
            times = [r[f"{crit}_time_s"] for r in self.results if r.get(f"{crit}_time_s") is not None]
            dist_key = f"{crit}_distance_m" if crit.startswith("sphere") else f"{crit}_horizontal_m"
            dists = [r[dist_key] for r in self.results if r.get(dist_key) is not None]
            summary[crit] = {
                "successes": successes,
                "success_rate": round(successes / n, 4),
                "success_rate_percentage": round(100 * successes / n, 2),
                "time_stats": self._stats(times),
                ("distance_stats" if crit.startswith("sphere") else "horizontal_stats"): self._stats(dists),
            }

        final_dists = [r["final_distance_m"] for r in self.results if r.get("final_distance_m") is not None]
        horiz = [r["horizontal_distance_m"] for r in self.results if r.get("horizontal_distance_m") is not None]
        vert = [r["vertical_distance_m"] for r in self.results if r.get("vertical_distance_m") is not None]
        times_all = [r["time_spent_s"] for r in self.results if r.get("time_spent_s") is not None]
        steps = [r["attack_steps"] for r in self.results if r.get("attack_steps") is not None]

        summary.update({
            "final_distance_stats": self._stats(final_dists),
            "horizontal_distance_stats": self._stats(horiz),
            "vertical_distance_stats": self._stats(vert),
            "time_stats": self._stats(times_all),
            "attack_step_stats": self._stats(steps),
            "total_duration_s": round(time.monotonic() - self.start_time, 2),
        })
        return summary

    def save(self, print_summary: bool = False) -> str:
        data = {"summary": self.get_summary(), "results": self.results}
        with open(self.log_file, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        if print_summary:
            self.print_summary()
        return self.log_file

    def print_summary(self) -> None:
        s = self.get_summary()
        n = s.get("total_tests", 0)
        if n == 0:
            print(f"[{self.test_name}] No results.")
            return
        print(f"\n[{self.test_name}] {n} episodes, duration {s.get('total_duration_s', 0):.1f}s")
        for crit in CRITERIA:
            cs = s[crit]
            print(f"  {crit:12s} {cs['success_rate_percentage']:5.1f}% ({cs['successes']}/{n})")
