#!/usr/bin/env python3
"""RQ6 / Figure 8 real-flight trajectories and injected bias. Supports claim C6 (paper Figure 8).

Usage:
    python3 -m analysis.generate_rq6_figure8 --source pre-baked
    python3 -m analysis.generate_rq6_figure8 --source pre-baked --bias-trial 3

Reads:
    results/<source>/rq6/real_evaluation/

Writes:
    analysis/figures/rq6_figure8b_trajectories_<source>.png
    analysis/figures/rq6_figure8c_bias_<source>.png

Notes:
    --bias-trial selects which real-flight trial is shown in the bias panel
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from analysis.common.geo import haversine_distance, latlon_to_meters
from analysis.common.rq6_ulog import select_logged_topics
from analysis.common.table_utils import default_results_dir, import_pyplot, sourced_figure_path
from analysis.common.ulog_utils import load_ulog

try:
    plt = import_pyplot()
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    import matplotlib.patheffects as path_effects
    from matplotlib.ticker import MultipleLocator
except ImportError:
    plt = None

SUCCESS_RADIUS_M = 10.0
BIAS_ZERO_THRESH = 1e-6

# Pohang field-test site (paper §7.6).
TARGET_LAT = 36.01351
TARGET_LON = 129.31921

_TRIAL_RE = re.compile(r"_trial(\d+)\.ulg$")

TRAJ_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]

def _attack_start_us(ulog):
    logged = select_logged_topics(ulog)
    bias = logged["bias"]
    if bias is None:
        return None, None
    ts = np.asarray(bias["timestamp"])
    mag = (
        np.abs(bias["gyro_bias_x"])
        + np.abs(bias["gyro_bias_y"])
        + np.abs(bias["gyro_bias_z"])
    )
    nz = np.where(mag > BIAS_ZERO_THRESH)[0]
    if nz.size == 0:
        return None, None
    return int(ts[nz[0]]), int(ts[nz[-1]])


def _heading_deg(ulog, t_us):
    logged = select_logged_topics(ulog)
    attitude = logged["attitude"]
    if attitude is None:
        return 0.0
    ts = np.asarray(attitude["timestamp"])
    i = int(np.argmin(np.abs(ts - t_us)))
    q0 = attitude["q[0]"][i]
    q1 = attitude["q[1]"][i]
    q2 = attitude["q[2]"][i]
    q3 = attitude["q[3]"][i]
    yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2),
                     1.0 - 2.0 * (q2 ** 2 + q3 ** 2))
    return float(np.degrees(yaw))


def _gps_segment(ulog, target_lat, target_lon, t_start_us):
    """GPS samples from attack start to first entry into the 10 m radius
    (or closest approach if it never reaches). `reach_us` is the raw ulog ts.
    """
    logged = select_logged_topics(ulog)
    gps = logged["gps"]
    if gps is None or logged["attitude"] is None or logged["imu"] is None:
        return None
    ts = np.asarray(gps["timestamp"])
    lats = np.asarray(gps["latitude_deg"])
    lons = np.asarray(gps["longitude_deg"])
    start_idx = int(np.argmin(np.abs(ts - t_start_us)))
    dists = haversine_distance(lats[start_idx:], lons[start_idx:], target_lat, target_lon)
    inside = np.where(dists <= SUCCESS_RADIUS_M)[0]
    end_idx = (start_idx + int(inside[0]) if inside.size
               else start_idx + int(np.argmin(dists)))
    return {
        "lat": lats[start_idx:end_idx + 1],
        "lon": lons[start_idx:end_idx + 1],
        "reach_us": int(ts[end_idx]),
        "reached": bool(inside.size),
    }


def _load_trial(path, target_lat, target_lon):
    ulog = load_ulog(path)
    t0_us, _ = _attack_start_us(ulog)
    if t0_us is None:
        return None
    heading = _heading_deg(ulog, t0_us)
    seg = _gps_segment(ulog, target_lat, target_lon, t0_us)
    if seg is None:
        return None
    east, north = latlon_to_meters(seg["lat"], seg["lon"], target_lat, target_lon)
    return {
        "east": east, "north": north,
        "heading_deg": heading,
        "reached": seg["reached"],
    }


def _load_bias(path, target_lat, target_lon):
    """Return (bias dict aligned to attack_start=0, success_rel_s)."""
    ulog = load_ulog(path)
    t0_us, _ = _attack_start_us(ulog)
    if t0_us is None:
        return None, None
    bias = select_logged_topics(ulog)["bias"]
    if bias is None:
        return None, None
    t_rel = (np.asarray(bias["timestamp"]) - t0_us) / 1e6

    seg = _gps_segment(ulog, target_lat, target_lon, t0_us)
    success_rel = (seg["reach_us"] - t0_us) / 1e6 if seg else float(t_rel[-1])

    return {
        "t":  t_rel,
        "bx": np.asarray(bias["gyro_bias_x"]),
        "by": np.asarray(bias["gyro_bias_y"]),
    }, float(success_rel)


def plot_trajectories(trials, output_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    all_x, all_y = [], []
    trial_handles = []

    for (trial_num, data), color in zip(trials, TRAJ_COLORS):
        if data is None:
            continue
        east, north = data["east"], data["north"]
        all_x.extend(east); all_y.extend(north)

        line, = ax.plot(east, north, color=color, linewidth=6, alpha=0.9, zorder=3)
        trial_handles.append((line, f"Trial {trial_num}"))

        arr_rot = 90.0 - data["heading_deg"]
        txt = ax.text(east[0], north[0], "\u27a4",
                      fontsize=50, color=color, ha="center", va="center",
                      rotation=arr_rot, fontweight="bold", zorder=6)
        txt.set_path_effects([path_effects.Stroke(linewidth=3.0, foreground="black"),
                              path_effects.Normal()])

        ax.scatter([east[-1]], [north[-1]], color=color, s=1000, marker="*",
                   edgecolors="black", linewidths=1.5, zorder=5)

    # Target X at origin
    ax.scatter([0], [0], color="red", s=1000, marker="X",
               edgecolors="black", linewidths=4, zorder=7)

    # 10 m radius circle
    theta = np.linspace(0, 2 * np.pi, 300)
    circle_x = SUCCESS_RADIUS_M * np.cos(theta)
    circle_y = SUCCESS_RADIUS_M * np.sin(theta)
    region_line, = ax.plot(circle_x, circle_y, "r--", alpha=0.55, linewidth=6.0)
    all_x.extend(circle_x); all_y.extend(circle_y)

    # Legend proxies
    arrow_proxy = ax.scatter([], [], marker="$\u27a4$", s=350, color="gray",
                             edgecolor="black", linewidth=1)
    target_proxy = ax.scatter([], [], color="red", s=250, marker="X",
                              edgecolors="black", linewidths=2)
    star_proxy = ax.scatter([], [], marker="*", s=300, color="gray",
                            edgecolors="black", linewidths=1.5)

    handles = [arrow_proxy, target_proxy, region_line, star_proxy]
    labels = ["Start Position & Heading", "Target",
              r"Target Region ($r=10$ m)", "Target Reached"]
    for h, lbl in trial_handles:
        handles.append(h); labels.append(lbl)

    ax.set_xlabel("Relative Position East (m)", fontsize=40, fontweight="bold")
    ax.set_ylabel("Relative Position North (m)", fontsize=40, fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    leg = ax.legend(handles, labels, fontsize=24, loc="upper left",
                    framealpha=0.5, ncols=1, handlelength=1.4, markerscale=1.5)
    leg.set_zorder(10)
    ax.tick_params(axis="both", which="major", labelsize=30)
    ax.set_aspect("equal", adjustable="datalim")

    if all_x:
        data_range = max(np.ptp(all_x), np.ptp(all_y))
        tick_interval = max(5.0, np.ceil(data_range / 6 / 5) * 5)
        ax.xaxis.set_major_locator(MultipleLocator(tick_interval))
        ax.yaxis.set_major_locator(MultipleLocator(tick_interval))

    plt.tight_layout()

    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 8(b) saved to {output_path}", file=sys.stderr)


def plot_bias(bias, success_rel, trial_num, output_path):
    if bias is None:
        print(f"No bias data for trial {trial_num}", file=sys.stderr)
        return

    t = bias["t"]; bx = bias["bx"]; by = bias["by"]

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.step(t, bx, where="post", linewidth=5, color="#1f77b4",
            label=r"$b_x$", alpha=0.9)
    ax.step(t, by, where="post", linewidth=5, color="#ff7f0e",
            label=r"$b_y$", alpha=0.9)
    ax.scatter(t, bx, color="#1f77b4", s=200, zorder=5,
               edgecolors="#0d3d5c", linewidths=3)
    ax.scatter(t, by, color="#ff7f0e", s=200, zorder=5,
               edgecolors="#cc6600", linewidths=3)

    ax.axhline(0.06, color="gray", linestyle=":", linewidth=3, alpha=0.7)
    ax.axhline(-0.06, color="gray", linestyle=":", linewidth=3, alpha=0.7)
    ax.axhline(0, color="gray", linestyle=":", linewidth=3, alpha=0.7)

    ax.axvline(0, color="red", linestyle="--", linewidth=4, alpha=1.0,
               label="Attack Start (0s)")
    if success_rel is not None and success_rel > float(t[0]):
        ax.axvline(success_rel, color="green", linestyle="--", linewidth=4,
                   alpha=1.0, label=f"Target Reached ({success_rel:.1f}s)")

    attack_end = float(t[-1])
    right = max(attack_end, success_rel or attack_end) + 2

    ax.set_xlabel("Time since Attack (s)", fontsize=40, fontweight="bold")
    ax.set_ylabel("Gyroscope Bias (rad/s)", fontsize=40, fontweight="bold")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=32, loc="upper center", framealpha=0.8,
              ncols=2, markerscale=1.0, columnspacing=0.65)
    ax.set_xlim(left=-2.0, right=right)
    ax.set_ylim([-0.075, 0.075])
    ax.tick_params(axis="both", which="major", labelsize=30)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 8(c) saved to {output_path}", file=sys.stderr)


def _collect(real_dir):
    out = []
    for f in sorted(real_dir.glob("*.ulg")):
        m = _TRIAL_RE.search(f.name)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description="Generate Figure 8")
    p.add_argument("--source", choices=["pre-baked", "fresh"], default="pre-baked")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--bias-trial", type=int, default=1,
                   help="Which trial to show in the bias panel (default 1)")
    args = p.parse_args()

    if plt is None:
        print("matplotlib required", file=sys.stderr)
        sys.exit(1)

    if args.results_dir:
        real_dir = Path(args.results_dir)
    else:
        real_dir = default_results_dir(args.source, "rq6") / "real_evaluation"

    trials = []
    bias_path = None
    for trial_num, f in _collect(real_dir):
        data = _load_trial(f, TARGET_LAT, TARGET_LON)
        trials.append((trial_num, data))
        if trial_num == args.bias_trial:
            bias_path = f

    if not trials:
        print(f"No trials found in {real_dir}", file=sys.stderr)
        return

    print(f"[rq6_figure8] Loaded {sum(1 for _, d in trials if d)}/{len(trials)} "
          f"trials from {real_dir}", file=sys.stderr)

    plot_trajectories(
        trials,
        sourced_figure_path("rq6_figure8b_trajectories", args.source, ".png"),
    )

    if bias_path:
        bias, success_rel = _load_bias(bias_path, TARGET_LAT, TARGET_LON)
        plot_bias(bias, success_rel, args.bias_trial,
                  sourced_figure_path("rq6_figure8c_bias", args.source, ".png"))


if __name__ == "__main__":
    main()
