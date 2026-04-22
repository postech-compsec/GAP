#!/usr/bin/env python3
"""Extract cropped per-attack CSVs from PX4 raw ULogs."""

from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from pyulog import ULog

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common.geo import haversine_distance, latlon_to_meters
from analysis.common.px4_attack_logs import (
    ATTACK_THRESHOLD,
    CSV_COLUMNS,
    SUCCESS_RADIUS_M,
    TARGET_LAT,
    TARGET_LON,
    WINDOW_POST_S,
    WINDOW_PRE_S,
    is_px4_attack_log_name,
    px4_attack_csv_root,
    px4_attack_raw_ulog_root,
    worker_of,
)

TOPICS = [
    "vehicle_global_position_groundtruth",
    "gyro_bias",
    "estimator_sensor_bias",
    "vehicle_status",
    "failsafe_flags",
]


def _dataset(ulog, name):
    for data in ulog.data_list:
        if data.name == name:
            return data.data
    return None


def _step_values(ts_src, values, ts_out, default):
    out = np.full(ts_out.shape, default, dtype=float)
    if ts_src is None or values is None or len(ts_src) == 0:
        return out
    idx = np.searchsorted(ts_src, ts_out, side="right") - 1
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return out


def _extract_one(path_str: str, out_root_str: str):
    path = Path(path_str)
    out_root = Path(out_root_str)
    worker = worker_of(path.name)
    if worker is not None:
        out_dir = out_root / f"worker{worker}"
    elif path.parent.name.startswith("worker"):
        out_dir = out_root / path.parent.name
    else:
        return {"file": path.name, "status": "error", "error": "cannot infer worker"}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.csv"

    try:
        ulog = ULog(str(path), TOPICS)
    except Exception as exc:
        return {"file": path.name, "status": "error", "error": str(exc)}

    pos = _dataset(ulog, "vehicle_global_position_groundtruth")
    bias = _dataset(ulog, "gyro_bias")
    if pos is None or bias is None:
        return {"file": path.name, "status": "error", "error": "missing required topic"}

    pos_ts = np.asarray(pos["timestamp"], dtype=np.int64)
    pos_lat = np.asarray(pos["lat"], dtype=float)
    pos_lon = np.asarray(pos["lon"], dtype=float)
    pos_alt = np.asarray(pos["alt"], dtype=float)
    pos_dist = np.asarray(haversine_distance(pos_lat, pos_lon, TARGET_LAT, TARGET_LON), dtype=float)

    bias_ts = np.asarray(bias["timestamp"], dtype=np.int64)
    inj_bx = np.asarray(bias["gyro_bias_x"], dtype=float)
    inj_by = np.asarray(bias["gyro_bias_y"], dtype=float)
    inj_bz = np.asarray(bias["gyro_bias_z"], dtype=float)
    attack_mag = np.abs(inj_bx) + np.abs(inj_by) + np.abs(inj_bz)
    attack_idx = np.where(attack_mag > ATTACK_THRESHOLD)[0]
    attack_start_us = int(bias_ts[attack_idx[0]]) if attack_idx.size else int(bias_ts[0])

    success_idx = np.where(pos_dist <= SUCCESS_RADIUS_M)[0]
    success_us = int(pos_ts[success_idx[0]]) if success_idx.size else None

    crop_start_us = attack_start_us - int(WINDOW_PRE_S * 1e6)
    tail_candidates = [int(pos_ts[-1]), int(bias_ts[-1])]

    ekf = _dataset(ulog, "estimator_sensor_bias")
    if ekf is not None:
        ekf_ts = np.asarray(ekf["timestamp"], dtype=np.int64)
        tail_candidates.append(int(ekf_ts[-1]))
    else:
        ekf_ts = None

    status = _dataset(ulog, "vehicle_status")
    if status is not None:
        status_ts = np.asarray(status["timestamp"], dtype=np.int64)
        tail_candidates.append(int(status_ts[-1]))
    else:
        status_ts = None

    flags = _dataset(ulog, "failsafe_flags")
    if flags is not None:
        flags_ts = np.asarray(flags["timestamp"], dtype=np.int64)
        tail_candidates.append(int(flags_ts[-1]))
    else:
        flags_ts = None

    crop_end_us = (
        success_us + int(WINDOW_POST_S * 1e6)
        if success_us is not None
        else max(tail_candidates)
    )

    streams = [
        pos_ts[(pos_ts >= crop_start_us) & (pos_ts <= crop_end_us)],
        bias_ts[(bias_ts >= crop_start_us) & (bias_ts <= crop_end_us)],
    ]
    if ekf_ts is not None:
        streams.append(ekf_ts[(ekf_ts >= crop_start_us) & (ekf_ts <= crop_end_us)])
    if status_ts is not None:
        streams.append(status_ts[(status_ts >= crop_start_us) & (status_ts <= crop_end_us)])
    if flags_ts is not None:
        streams.append(flags_ts[(flags_ts >= crop_start_us) & (flags_ts <= crop_end_us)])

    timestamps = np.unique(
        np.concatenate([
            np.asarray([crop_start_us, crop_end_us], dtype=np.int64),
            *[stream for stream in streams if len(stream) > 0],
        ])
    )
    timestamps.sort()

    lat = _step_values(pos_ts, pos_lat, timestamps, np.nan)
    lon = _step_values(pos_ts, pos_lon, timestamps, np.nan)
    alt = _step_values(pos_ts, pos_alt, timestamps, np.nan)
    x_m, y_m = latlon_to_meters(lat, lon, TARGET_LAT, TARGET_LON)
    dist_m = haversine_distance(lat, lon, TARGET_LAT, TARGET_LON)
    reached = np.isfinite(dist_m) & (dist_m <= SUCCESS_RADIUS_M)

    position_sample = np.isin(timestamps, pos_ts).astype(int)
    injected_bias_sample = np.isin(timestamps, bias_ts).astype(int)
    ekf_bias_sample = np.isin(timestamps, ekf_ts).astype(int) if ekf_ts is not None else np.zeros(timestamps.shape, dtype=int)
    vehicle_status_sample = np.isin(timestamps, status_ts).astype(int) if status_ts is not None else np.zeros(timestamps.shape, dtype=int)
    failsafe_flags_sample = np.isin(timestamps, flags_ts).astype(int) if flags_ts is not None else np.zeros(timestamps.shape, dtype=int)

    ekf_bx = ekf_by = ekf_bz = np.full(timestamps.shape, np.nan, dtype=float)
    if ekf is not None:
        ekf_bx = _step_values(ekf_ts, np.asarray(ekf["gyro_bias[0]"], dtype=float), timestamps, np.nan)
        ekf_by = _step_values(ekf_ts, np.asarray(ekf["gyro_bias[1]"], dtype=float), timestamps, np.nan)
        ekf_bz = _step_values(ekf_ts, np.asarray(ekf["gyro_bias[2]"], dtype=float), timestamps, np.nan)

    failsafe = np.zeros(timestamps.shape, dtype=int)
    if status is not None and "failsafe" in status:
        failsafe = _step_values(status_ts, np.asarray(status["failsafe"], dtype=int), timestamps, 0).astype(int)

    angular_velocity_invalid = np.zeros(timestamps.shape, dtype=int)
    if flags is not None and "angular_velocity_invalid" in flags:
        angular_velocity_invalid = _step_values(
            flags_ts,
            np.asarray(flags["angular_velocity_invalid"], dtype=int),
            timestamps,
            0,
        ).astype(int)

    t_rel_s = (timestamps - attack_start_us) / 1e6
    injected_bx = _step_values(bias_ts, inj_bx, timestamps, 0.0)
    injected_by = _step_values(bias_ts, inj_by, timestamps, 0.0)
    injected_bz = _step_values(bias_ts, inj_bz, timestamps, 0.0)

    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in zip(
            timestamps,
            t_rel_s,
            lat,
            lon,
            alt,
            x_m,
            y_m,
            dist_m,
            reached.astype(int),
            injected_bx,
            injected_by,
            injected_bz,
            ekf_bx,
            ekf_by,
            ekf_bz,
            failsafe,
            angular_velocity_invalid,
            position_sample,
            injected_bias_sample,
            ekf_bias_sample,
            vehicle_status_sample,
            failsafe_flags_sample,
        ):
            writer.writerow([
                int(row[0]),
                f"{float(row[1]):.6f}",
                f"{float(row[2]):.8f}" if np.isfinite(row[2]) else "",
                f"{float(row[3]):.8f}" if np.isfinite(row[3]) else "",
                f"{float(row[4]):.6f}" if np.isfinite(row[4]) else "",
                f"{float(row[5]):.4f}" if np.isfinite(row[5]) else "",
                f"{float(row[6]):.4f}" if np.isfinite(row[6]) else "",
                f"{float(row[7]):.4f}" if np.isfinite(row[7]) else "",
                int(row[8]),
                f"{float(row[9]):.6f}",
                f"{float(row[10]):.6f}",
                f"{float(row[11]):.6f}",
                f"{float(row[12]):.6f}" if np.isfinite(row[12]) else "",
                f"{float(row[13]):.6f}" if np.isfinite(row[13]) else "",
                f"{float(row[14]):.6f}" if np.isfinite(row[14]) else "",
                int(row[15]),
                int(row[16]),
                int(row[17]),
                int(row[18]),
                int(row[19]),
                int(row[20]),
                int(row[21]),
            ])

    return {
        "file": path.name,
        "status": "ok",
        "rows": len(timestamps),
        "success": success_us is not None,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract cropped PX4 attack CSVs")
    parser.add_argument(
        "--src",
        type=Path,
        default=px4_attack_raw_ulog_root("pre-baked"),
        help="Source directory containing raw PX4 ULogs",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=px4_attack_csv_root("pre-baked"),
        help="Destination directory for extracted CSVs",
    )
    args = parser.parse_args()

    paths = (
        sorted(path for path in args.src.rglob("*.ulg") if is_px4_attack_log_name(path.name))
        if args.src.exists() else []
    )
    if not paths:
        print(f"No ULogs found under {args.src}", file=sys.stderr)
        return

    args.dst.mkdir(parents=True, exist_ok=True)
    jobs = cpu_count() or 1
    print(f"Extracting {len(paths)} ULogs -> {args.dst} on {jobs} workers", file=sys.stderr)

    with Pool(processes=jobs) as pool:
        results = pool.starmap(_extract_one, [(str(path), str(args.dst)) for path in paths])

    ok = sum(1 for result in results if result["status"] == "ok")
    errors = [result for result in results if result["status"] != "ok"]
    print(f"Extracted {ok}/{len(results)} CSVs", file=sys.stderr)
    if errors:
        print(f"{len(errors)} errors:", file=sys.stderr)
        for result in errors[:10]:
            print(f"  {result['file']}: {result['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
