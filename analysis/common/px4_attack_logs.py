"""Shared PX4 attack-flight CSV/ULog helpers used by RQ1 Figure 7 and RQ5."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from analysis.common.table_utils import project_root

TARGET_LAT = 47.39855040647849
TARGET_LON = 8.545290332727657
SUCCESS_RADIUS_M = 10.0
ATTACK_THRESHOLD = 1e-6
WINDOW_PRE_S = 10.0
WINDOW_POST_S = 10.0

CSV_COLUMNS = [
    "timestamp_us",
    "t_rel_s",
    "lat",
    "lon",
    "alt_m",
    "x_m",
    "y_m",
    "dist_target_m",
    "reached_target",
    "injected_bx_rad_s",
    "injected_by_rad_s",
    "injected_bz_rad_s",
    "ekf_bx_rad_s",
    "ekf_by_rad_s",
    "ekf_bz_rad_s",
    "failsafe_active",
    "angular_velocity_invalid",
    "position_sample",
    "injected_bias_sample",
    "ekf_bias_sample",
    "vehicle_status_sample",
    "failsafe_flags_sample",
]

_WORKER_RE = re.compile(r"_w(\d{1,2})_")


def _shared_px4_root(source: str) -> Path:
    return project_root() / "results" / source / "flight-logs" / "px4"


def px4_attack_csv_root(source: str) -> Path:
    return _shared_px4_root(source) / "csv"


def px4_attack_raw_ulog_root(source: str) -> Path:
    return _shared_px4_root(source) / "raw"


def is_px4_attack_log_name(name: str) -> bool:
    # `gap-rl` is a legacy prefix kept for shipped pre-baked PX4 corpora.
    # Newer runs should use the unified `gap_px4-jmavsim` prefix.
    return (
        "_gap_px4-jmavsim_" in name
        or "_gap-rl_px4-jmavsim_" in name
    )


def iter_px4_attack_csvs(source: str) -> list[Path]:
    root = px4_attack_csv_root(source)
    return sorted(
        path
        for path in root.glob("worker*/*.csv")
        if path.is_file() and is_px4_attack_log_name(path.name)
    )


def iter_px4_attack_ulogs(source: str) -> list[Path]:
    root = px4_attack_raw_ulog_root(source)
    return (
        sorted(path for path in root.rglob("*.ulg") if is_px4_attack_log_name(path.name))
        if root.exists() else []
    )


def _latest_mtime(paths: list[Path]) -> float:
    return max((path.stat().st_mtime for path in paths), default=0.0)


def _needs_csv_refresh(source: str, csv_paths: list[Path], ulog_paths: list[Path]) -> bool:
    if not ulog_paths:
        return False
    if not csv_paths:
        return True
    if source != "fresh":
        return False
    csv_stems = {path.stem for path in csv_paths}
    ulog_stems = {path.stem for path in ulog_paths}
    if ulog_stems - csv_stems:
        return True
    return _latest_mtime(ulog_paths) > _latest_mtime(csv_paths)


def refresh_px4_attack_csvs(source: str, *, verbose: bool = False) -> list[Path]:
    script = project_root() / "src" / "tools" / "extract_px4_attack_csvs.py"
    cmd = [
        sys.executable,
        str(script),
        "--src",
        str(px4_attack_raw_ulog_root(source)),
        "--dst",
        str(px4_attack_csv_root(source)),
    ]
    if verbose:
        print(
            f"[px4_attack_logs] Extracting worker CSVs from raw ULogs ({source})",
            file=sys.stderr,
        )
    subprocess.run(cmd, check=True)
    return iter_px4_attack_csvs(source)


def prepare_px4_attack_csvs(
    source: str,
    *,
    force_refresh: bool = False,
    verbose: bool = False,
) -> list[Path]:
    csv_paths = iter_px4_attack_csvs(source)
    ulog_paths = iter_px4_attack_ulogs(source)
    if force_refresh or _needs_csv_refresh(source, csv_paths, ulog_paths):
        if verbose:
            if force_refresh:
                print(
                    f"[px4_attack_logs] Forced refresh from raw ULogs ({source})",
                    file=sys.stderr,
                )
            elif source == "fresh":
                print(
                    f"[px4_attack_logs] Fresh source: prioritizing raw ULogs over cached CSVs",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[px4_attack_logs] No attack CSVs found; extracting from raw ULogs ({source})",
                    file=sys.stderr,
                )
        csv_paths = refresh_px4_attack_csvs(source, verbose=False)
    return csv_paths


def worker_of(name: str) -> int | None:
    match = _WORKER_RE.search(name)
    return int(match.group(1)) if match else None


def load_px4_attack_csv(path: Path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if getattr(data, "size", 0) == 0:
        raise ValueError(f"{path} is empty")
    if data.shape == ():
        data = data.reshape(1)
    return data


def _sample_mask(data, field: str):
    if field in (data.dtype.names or ()):
        return np.asarray(data[field], dtype=int) > 0
    return np.ones(len(data), dtype=bool)


def success_time_s(data) -> float | None:
    reached = np.asarray(data["reached_target"], dtype=int) > 0
    reached &= _sample_mask(data, "position_sample")
    if not np.any(reached):
        return None
    return float(np.asarray(data["t_rel_s"], dtype=float)[np.where(reached)[0][0]])


def evaluation_mask(data):
    t_rel = np.asarray(data["t_rel_s"], dtype=float)
    mask = t_rel >= 0.0
    success_t = success_time_s(data)
    if success_t is not None:
        mask &= t_rel <= success_t + 1e-9
    return mask


def deactivation_times_s(data) -> list[float]:
    t_rel = np.asarray(data["t_rel_s"], dtype=float)
    active = np.asarray(data["failsafe_active"], dtype=int) > 0
    mask = evaluation_mask(data)
    mask &= _sample_mask(data, "vehicle_status_sample")
    if not np.any(mask):
        return []

    times = []
    prev = False
    start = None
    for t, value in zip(t_rel[mask], active[mask]):
        value = bool(value)
        if value and not prev:
            start = float(t)
        elif not value and prev and start is not None:
            times.append(float(t - start))
            start = None
        prev = value
    if prev and start is not None:
        times.append(float(t_rel[mask][-1] - start))
    return times


def summarize_px4_attack_csv(path: Path) -> dict:
    data = load_px4_attack_csv(path)
    mask = evaluation_mask(data)
    gyro = np.asarray(data["angular_velocity_invalid"], dtype=int) > 0
    failsafe = np.asarray(data["failsafe_active"], dtype=int) > 0
    gyro_mask = mask & _sample_mask(data, "failsafe_flags_sample")
    failsafe_mask = mask & _sample_mask(data, "vehicle_status_sample")
    success_t = success_time_s(data)
    return {
        "file": path.name,
        "status": "ok" if success_t is not None else "skip",
        "gyro_triggered": bool(np.any(gyro[gyro_mask])) if np.any(gyro_mask) else False,
        "failsafe_triggered": bool(np.any(failsafe[failsafe_mask])) if np.any(failsafe_mask) else False,
        "deactivation_times_s": deactivation_times_s(data),
    }


def summarize_px4_attack_csvs(paths: list[Path]) -> dict:
    records = [summarize_px4_attack_csv(path) for path in paths]
    ok = [record for record in records if record["status"] == "ok"]
    deactivation_times = [t for record in ok for t in record["deactivation_times_s"]]
    gyro_count = sum(1 for record in ok if record["gyro_triggered"])
    failsafe_count = sum(1 for record in ok if record["failsafe_triggered"])

    return {
        "total_ulogs": len(records),
        "successful_attack_episodes": len(ok),
        "gyro_failsafe_triggered_count": gyro_count,
        "gyro_failsafe_triggered_pct": round(100.0 * gyro_count / len(ok), 2) if ok else 0.0,
        "other_failsafe_triggered_count": failsafe_count,
        "other_failsafe_triggered_pct": round(100.0 * failsafe_count / len(ok), 2) if ok else 0.0,
        "avg_deactivation_time_s": round(sum(deactivation_times) / len(deactivation_times), 3) if deactivation_times else None,
        "max_deactivation_time_s": round(max(deactivation_times), 3) if deactivation_times else None,
    }
