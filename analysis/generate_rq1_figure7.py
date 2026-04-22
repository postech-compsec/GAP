#!/usr/bin/env python3
"""RQ1 / Figure 7 successful-attack trajectory and bias analysis. Supports claim C1 (paper Figure 7).

Usage:
    python3 -m analysis.generate_rq1_figure7 --source pre-baked
    python3 -m analysis.generate_rq1_figure7 --source pre-baked --use-ulogs

Reads:
    results/<source>/flight-logs/px4/csv/ by default
    results/<source>/flight-logs/px4/raw/ with --use-ulogs

Writes:
    analysis/figures/rq1_figure7a_trajectories_<source>.png
    analysis/figures/rq1_figure7b_bias_<source>.png
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from analysis.common.px4_attack_logs import (
    SUCCESS_RADIUS_M,
    TARGET_LAT,
    TARGET_LON,
    iter_px4_attack_csvs,
    iter_px4_attack_ulogs,
    load_px4_attack_csv,
    px4_attack_csv_root,
    px4_attack_raw_ulog_root,
    worker_of,
)
from analysis.common.table_utils import import_pyplot, sourced_figure_path
from analysis.common.ulog_utils import load_ulog

try:
    plt = import_pyplot()
    import matplotlib

    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as path_effects
except ImportError:
    plt = None

DEFAULT_SEED = 0


def _ds(ulog, name):
    for d in ulog.data_list:
        if d.name == name:
            return d.data
    return None


def _latlon_to_meters(lat, lon, ref_lat, ref_lon):
    R = 6371000.0
    x = R * np.radians(lon - ref_lon) * np.cos(np.radians(ref_lat))
    y = R * np.radians(lat - ref_lat)
    return x, y


def _trajectory_from_ulog(path):
    try:
        ulog = load_ulog(path)
    except Exception:
        return None

    gt = _ds(ulog, "vehicle_global_position_groundtruth") or _ds(ulog, "vehicle_global_position")
    if gt is not None and "lat" in gt and "lon" in gt:
        lat = np.asarray(gt["lat"])
        lon = np.asarray(gt["lon"])
        x, y = _latlon_to_meters(lat, lon, TARGET_LAT, TARGET_LON)
    else:
        lp = _ds(ulog, "vehicle_local_position")
        if lp is None:
            return None
        x = np.asarray(lp["x"])
        y = np.asarray(lp["y"])

    dist = np.sqrt(x * x + y * y)
    if not np.any(dist <= SUCCESS_RADIUS_M):
        return None
    target_idx = int(np.argmax(dist <= SUCCESS_RADIUS_M))
    return {"x": x, "y": y, "target_idx": target_idx, "path": path}


def _bias_record_from_ulog(path):
    """Return injected/estimated bias series aligned to attack start."""
    ulog = load_ulog(path)
    bias = _ds(ulog, "gyro_bias")
    ekf = _ds(ulog, "estimator_sensor_bias")
    gt = _ds(ulog, "vehicle_global_position_groundtruth") or _ds(ulog, "vehicle_global_position")
    if bias is None or gt is None:
        return None, None, None

    bias_ts = np.asarray(bias["timestamp"], dtype=np.int64)
    bx_all = np.asarray(bias["gyro_bias_x"])
    by_all = np.asarray(bias["gyro_bias_y"])
    bz_all = np.asarray(bias["gyro_bias_z"])
    mag = np.abs(bx_all) + np.abs(by_all) + np.abs(bz_all)
    nz = np.where(mag > 1e-6)[0]
    if nz.size == 0:
        return None, None, None
    t0_us = int(bias_ts[nz[0]])

    keep = slice(int(nz[0]), None)
    bias_out = {
        "t": (bias_ts[keep] - t0_us) / 1e6,
        "bx": bx_all[keep],
        "by": by_all[keep],
    }

    ekf_out = None
    if ekf is not None:
        ekf_out = {
            "t": (np.asarray(ekf["timestamp"], dtype=np.int64) - t0_us) / 1e6,
            "bx": np.asarray(ekf["gyro_bias[0]"]),
            "by": np.asarray(ekf["gyro_bias[1]"]),
        }

    gt_t_rel = (np.asarray(gt["timestamp"], dtype=np.int64) - t0_us) / 1e6
    gx, gy = _latlon_to_meters(
        np.asarray(gt["lat"]),
        np.asarray(gt["lon"]),
        TARGET_LAT,
        TARGET_LON,
    )
    dist = np.sqrt(gx * gx + gy * gy)
    post = gt_t_rel >= 0
    gt_t_rel = gt_t_rel[post]
    dist = dist[post]
    if gt_t_rel.size == 0:
        return None
    reach = np.where(dist <= SUCCESS_RADIUS_M)[0]
    success_rel = float(gt_t_rel[int(reach[0])] if reach.size else gt_t_rel[-1])
    record = {
        "file": Path(path).name,
        "success_rel_s": round(float(success_rel), 6),
        "bias": {
            "t": _round_list(bias_out["t"]),
            "bx": _round_list(bias_out["bx"]),
            "by": _round_list(bias_out["by"]),
        },
        "ekf": None,
    }
    if ekf_out is not None:
        record["ekf"] = {
            "t": _round_list(ekf_out["t"]),
            "bx": _round_list(ekf_out["bx"]),
            "by": _round_list(ekf_out["by"]),
        }
    return record


def _downsample_indices(length: int, max_points: int):
    if length <= max_points:
        return np.arange(length, dtype=int)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=int))


def _round_list(values, digits=6):
    return [round(float(v), digits) for v in values]


def _sample_mask(data, field: str):
    if field in (data.dtype.names or ()):
        return np.asarray(data[field], dtype=int) > 0
    return np.ones(len(data), dtype=bool)


def _group_paths_by_worker(paths):
    groups = {i: [] for i in range(1, 13)}
    for path in sorted(paths):
        worker = worker_of(path.name)
        if worker is not None:
            groups[worker].append(path)
    return groups


def _sample(groups, trajectory_loader, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    picks = {}
    for worker in range(1, 13):
        paths = groups[worker][:]
        if not paths:
            continue
        rng.shuffle(paths)
        for path in paths:
            traj = trajectory_loader(path)
            if traj is None:
                continue
            traj["worker"] = worker
            picks[worker] = traj
            break
    return picks


def _trajectory_record(traj, max_points=600):
    end = traj["target_idx"] + 1
    x = np.asarray(traj["x"][:end])
    y = np.asarray(traj["y"][:end])
    keep = _downsample_indices(len(x), max_points)
    x = x[keep]
    y = y[keep]
    return {
        "worker": int(traj["worker"]),
        "file": Path(traj["path"]).name,
        "x": _round_list(x, 4),
        "y": _round_list(y, 4),
    }


def _trajectory_from_csv(path):
    try:
        data = load_px4_attack_csv(path)
    except Exception:
        return None

    x = np.asarray(data["x_m"], dtype=float)
    y = np.asarray(data["y_m"], dtype=float)
    reached = np.asarray(data["reached_target"], dtype=int) > 0
    valid = np.isfinite(x) & np.isfinite(y) & _sample_mask(data, "position_sample")
    x = x[valid]
    y = y[valid]
    reached = reached[valid]
    if not np.any(reached):
        return None
    target_idx = int(np.where(reached)[0][0])
    return {"x": x, "y": y, "target_idx": target_idx, "path": path}


def _bias_record_from_csv(path):
    try:
        data = load_px4_attack_csv(path)
    except Exception:
        return None

    t = np.asarray(data["t_rel_s"], dtype=float)
    reached = np.asarray(data["reached_target"], dtype=int) > 0
    reached &= _sample_mask(data, "position_sample")
    success_idx = np.where(reached)[0]
    if success_idx.size == 0:
        return None
    success_rel = float(t[success_idx[0]])

    bias_t = t
    bias_bx = np.asarray(data["injected_bx_rad_s"], dtype=float)
    bias_by = np.asarray(data["injected_by_rad_s"], dtype=float)
    bias_mask = np.isfinite(bias_t) & np.isfinite(bias_bx) & np.isfinite(bias_by)
    bias_mask &= _sample_mask(data, "injected_bias_sample")

    ekf_t = t
    ekf_bx = np.asarray(data["ekf_bx_rad_s"], dtype=float)
    ekf_by = np.asarray(data["ekf_by_rad_s"], dtype=float)
    ekf_mask = np.isfinite(ekf_t) & np.isfinite(ekf_bx) & np.isfinite(ekf_by)
    ekf_mask &= _sample_mask(data, "ekf_bias_sample")

    record = {
        "file": Path(path).name,
        "success_rel_s": round(success_rel, 6),
        "bias": {
            "t": _round_list(bias_t[bias_mask]),
            "bx": _round_list(bias_bx[bias_mask]),
            "by": _round_list(bias_by[bias_mask]),
        },
        "ekf": None,
    }
    if np.any(ekf_mask):
        record["ekf"] = {
            "t": _round_list(ekf_t[ekf_mask]),
            "bx": _round_list(ekf_bx[ekf_mask]),
            "by": _round_list(ekf_by[ekf_mask]),
        }
    return record


def _build_summary(paths, trajectory_loader, bias_loader, seed: int = DEFAULT_SEED):
    groups = _group_paths_by_worker(paths)
    picks = _sample(groups, trajectory_loader, seed=seed)
    if not picks:
        return None

    trajectory_samples = [_trajectory_record(picks[w]) for w in sorted(picks)]
    rng = random.Random(seed + 1)
    bias_path = rng.choice(list(picks.values()))["path"]
    bias_sample = bias_loader(bias_path)

    return {
        "seed": seed,
        "success_radius_m": SUCCESS_RADIUS_M,
        "trajectory_samples": trajectory_samples,
        "bias_sample": bias_sample,
    }


def plot_trajectories(records, output_path):
    fig, ax = plt.subplots(figsize=(14, 14))

    all_colors = list(mcolors.TABLEAU_COLORS.values())
    success_colors = [c for c in all_colors if c not in ("tab:red", "tab:pink")]
    if len(success_colors) < len(records):
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        success_colors = [cmap[i] for i in range(20) if i not in (6, 7)]

    for i, traj in enumerate(sorted(records, key=lambda record: record["worker"])):
        x = np.asarray(traj["x"])
        y = np.asarray(traj["y"])
        color = success_colors[i % len(success_colors)]

        line, = ax.plot(x, y, color=color, alpha=0.85, linewidth=5.0)
        line.set_path_effects([
            path_effects.Stroke(linewidth=7.0, foreground="black"),
            path_effects.Normal(),
        ])

        txt = ax.text(
            x[0],
            y[0],
            "\u27a4",
            fontsize=80,
            color=color,
            ha="center",
            va="center",
            rotation=90,
            fontweight="bold",
            zorder=7,
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="black"),
            path_effects.Normal(),
        ])

        if i == 0:
            ax.scatter(
                [],
                [],
                color="gray",
                s=400,
                marker="$\u27a4$",
                edgecolor="black",
                linewidth=2,
                label="Start Position & Heading (220m)",
            )

    ax.add_patch(plt.Circle((0, 0), 220, color="gray", fill=False, linewidth=3, linestyle="-", alpha=0.8))
    ax.add_patch(
        plt.Circle(
            (0, 0),
            SUCCESS_RADIUS_M,
            color="black",
            fill=True,
            linewidth=5,
            linestyle="-",
            label=r"Target Region ($r=10$m)",
            zorder=8,
        )
    )
    flag = ax.text(0, -1, "\u2691", fontsize=30, color="red", ha="center", va="center", fontweight="bold", zorder=8)
    flag.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground="black"),
        path_effects.Normal(),
    ])
    ax.plot([], [], color="gray", linewidth=5.0, label="Successful Trajectories")

    ax.set_aspect("equal")
    ax.set_xlim([-240, 240])
    ax.set_ylim([-240, 240])
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("East (meters)", fontsize=24, fontweight="bold")
    ax.set_ylabel("North (meters)", fontsize=24, fontweight="bold")

    legend = ax.legend(loc="upper left", fontsize=20, framealpha=0.75)
    legend.set_zorder(10)
    ax.tick_params(axis="both", which="major", labelsize=30)

    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)
    fig.savefig(str(output_path), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"RQ1 Figure 7(a) saved to {output_path}", file=sys.stderr)


def plot_bias(bias_sample, output_path):
    bias = bias_sample["bias"]
    ekf = bias_sample.get("ekf")
    success_rel = float(bias_sample["success_rel_s"])

    bt = np.asarray(bias["t"])
    bx = np.asarray(bias["bx"])
    by = np.asarray(bias["by"])
    bias_mask = bt <= success_rel
    bt = bt[bias_mask]
    bx = bx[bias_mask]
    by = by[bias_mask]

    fig, ax = plt.subplots(figsize=(24, 9))

    ax.step(bt, bx, where="post", label=r"$b_x$ (Injected)", color="#1f77b4", linewidth=4)
    ax.step(bt, by, where="post", label=r"$b_y$ (Injected)", color="#ff7f0e", linewidth=4)

    if ekf is not None:
        et = np.asarray(ekf["t"])
        ekf_mask = et <= success_rel
        ax.plot(et[ekf_mask], np.asarray(ekf["bx"])[ekf_mask], label=r"$\hat{b}_x$ (Estimated)", color="#2ca02c", linewidth=4)
        ax.plot(et[ekf_mask], np.asarray(ekf["by"])[ekf_mask], label=r"$\hat{b}_y$ (Estimated)", color="#d62728", linewidth=4)

    ax.axhline(0.06, color="gray", linestyle=":", linewidth=2, alpha=0.7)
    ax.axhline(-0.06, color="gray", linestyle=":", linewidth=2, alpha=0.7)
    ax.axhline(0, color="gray", linestyle=":", linewidth=2, alpha=0.7)

    ax.axvline(0, color="red", linestyle="--", linewidth=3, alpha=0.9, label="Attack Start (0s)")
    ax.axvline(success_rel, color="green", linestyle="--", linewidth=3, alpha=0.9, label=f"Target Reached ({success_rel:.1f}s)")

    ax.set_xlabel("Time since Attack (s)", fontsize=40, fontweight="bold")
    ax.set_ylabel("Gyroscope Bias (rad/s)", fontsize=40, fontweight="bold")
    ax.legend(loc="upper center", fontsize=32, framealpha=0.8, ncols=3, columnspacing=0.65)
    ax.grid(True, alpha=0.3, linestyle="--")

    right_edge = max(float(bt[-1]), success_rel)
    time_range = right_edge - float(bt[0]) if len(bt) > 1 else 1.0
    margin = time_range * 0.05
    ax.set_xlim([float(bt[0]) - margin, right_edge + margin])
    ax.set_ylim([-0.075, 0.075])
    ax.tick_params(axis="both", which="major", labelsize=32)

    plt.tight_layout()
    fig.savefig(str(output_path), format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"RQ1 Figure 7(b) saved to {output_path}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description="Generate RQ1 Figure 7")
    p.add_argument("--source", choices=["pre-baked", "fresh"], default="pre-baked")
    p.add_argument("--use-ulogs", action="store_true",
                   help="Read raw ULogs instead of the extracted worker CSV corpus.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    if plt is None:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    use_ulogs = args.use_ulogs
    paths = iter_px4_attack_ulogs(args.source) if use_ulogs else iter_px4_attack_csvs(args.source)
    if not use_ulogs and not paths:
        raw_paths = iter_px4_attack_ulogs(args.source)
        if raw_paths:
            use_ulogs = True
            paths = raw_paths
            print(f"[rq1_figure7] No extracted CSVs found; reading raw shared attack ULogs from {px4_attack_raw_ulog_root(args.source)}", file=sys.stderr)

    if use_ulogs:
        if args.use_ulogs:
            print(f"[rq1_figure7] Reading raw shared attack ULogs from {px4_attack_raw_ulog_root(args.source)}", file=sys.stderr)
        summary = _build_summary(paths, _trajectory_from_ulog, _bias_record_from_ulog, seed=args.seed)
    else:
        print(f"[rq1_figure7] Reading shared attack CSVs from {px4_attack_csv_root(args.source)}", file=sys.stderr)
        summary = _build_summary(paths, _trajectory_from_csv, _bias_record_from_csv, seed=args.seed)
    if summary is None:
        print("[rq1_figure7] No successful shared attack samples found", file=sys.stderr)
        return

    trajectories = summary.get("trajectory_samples", [])
    if trajectories:
        plot_trajectories(
            trajectories,
            sourced_figure_path("rq1_figure7a_trajectories", args.source, ".png"),
        )

    bias_sample = summary.get("bias_sample")
    if bias_sample is not None:
        plot_bias(
            bias_sample,
            sourced_figure_path("rq1_figure7b_bias", args.source, ".png"),
        )


if __name__ == "__main__":
    main()
