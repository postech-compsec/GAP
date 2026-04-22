#!/usr/bin/env python3
"""RQ6 / Table 6 per-trial metrics for the real-world flights. Supports claim C6 (paper Table 6).

Usage:
    python3 -m analysis.generate_rq6_table6 --source pre-baked

Reads:
    results/<source>/rq6/real_evaluation/

Writes:
    analysis/csv/rq6_table6_<source>.csv
    analysis/figures/rq6_table6_<source>.png

Notes:
    post-analysis uses the logged ULog equivalents of the runtime
    GPS_RAW_INT, ATTITUDE_QUATERNION, SCALED_IMU, and gyro-bias streams
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from analysis.common.geo import haversine_distance
from analysis.common.table_utils import (
    default_results_dir,
    emit_csv,
    figures_dir,
    import_pyplot,
    sourced_csv_path,
    sourced_figure_path,
)
from analysis.common.rq6_ulog import select_logged_topics
from analysis.common.ulog_utils import load_ulog

SUCCESS_RADIUS_M = 10.0

# Pohang field-test site (paper §7.6).
TARGET_LAT = 36.01351
TARGET_LON = 129.31921

# Paper-sourced per-trial annotations (not in the ulg).
TRIAL_METADATA = {
    1: {"airframe": "X500", "wind_mps": "1.7 to 2.1", "gusts_mps": "2.8 to 3.1"},
    2: {"airframe": "S500", "wind_mps": "1.9 to 2.8", "gusts_mps": "2.8 to 3.2"},
    3: {"airframe": "S500", "wind_mps": "3.4 to 3.9", "gusts_mps": "4.5 to 6.4"},
    4: {"airframe": "S500", "wind_mps": "3.4 to 4.4", "gusts_mps": "4.7 to 7.9"},
    5: {"airframe": "S500", "wind_mps": "3.4 to 4.4", "gusts_mps": "4.7 to 7.9"},
    6: {"airframe": "S500", "wind_mps": "3.4 to 4.4", "gusts_mps": "4.7 to 7.9"},
    7: {"airframe": "S500", "wind_mps": "3.4 to 4.4", "gusts_mps": "4.7 to 7.9"},
}
BIAS_ZERO_THRESH = 1e-6
_TRIAL_RE = re.compile(r"_trial(\d+)\.ulg$")


def _find_attack_window(ulog):
    """(first_nonzero_bias_s, last_nonzero_bias_s) or (None, None)."""
    logged = select_logged_topics(ulog)
    bias = logged["bias"]
    if bias is None:
        return None, None
    ts = np.asarray(bias["timestamp"]) / 1e6
    magnitude = (
        np.abs(bias["gyro_bias_x"])
        + np.abs(bias["gyro_bias_y"])
        + np.abs(bias["gyro_bias_z"])
    )
    nz = np.where(magnitude > BIAS_ZERO_THRESH)[0]
    if nz.size == 0:
        return None, None
    return float(ts[nz[0]]), float(ts[nz[-1]])


def _gps_segment(ulog, target_lat, target_lon, t_start):
    """GPS samples from t_start to first entry into the 10 m radius (or
    closest approach if it never enters)."""
    logged = select_logged_topics(ulog)
    if logged["gps"] is None or logged["attitude"] is None or logged["imu"] is None:
        return None

    gps = logged["gps"]
    ts = np.asarray(gps["timestamp"]) / 1e6
    lats = np.asarray(gps["latitude_deg"])
    lons = np.asarray(gps["longitude_deg"])

    start_idx = int(np.argmin(np.abs(ts - t_start)))
    dists = haversine_distance(lats[start_idx:], lons[start_idx:], target_lat, target_lon)
    inside = np.where(dists <= SUCCESS_RADIUS_M)[0]
    if inside.size > 0:
        end_idx = start_idx + int(inside[0])
        reached = True
    else:
        end_idx = start_idx + int(np.argmin(dists))
        reached = False
    return {
        "t": ts[start_idx:end_idx + 1] - t_start,
        "lat": lats[start_idx:end_idx + 1],
        "lon": lons[start_idx:end_idx + 1],
        "reached": reached,
    }


def _traj_length_m(lat, lon):
    if lat.size < 2:
        return 0.0
    segs = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return float(np.sum(segs))


def analyze_ulog(path, target_lat, target_lon):
    ulog = load_ulog(path)
    t0, _ = _find_attack_window(ulog)
    if t0 is None:
        return None

    seg = _gps_segment(ulog, target_lat, target_lon, t0)
    if seg is None:
        return None

    init_dist = float(haversine_distance(seg["lat"][0], seg["lon"][0],
                                         target_lat, target_lon))
    time_to = float(seg["t"][-1])
    traj_len = _traj_length_m(seg["lat"], seg["lon"])
    return {
        "init_dist_m": init_dist,
        "time_s": time_to,
        "traj_length_m": traj_len,
        "reached": seg["reached"],
        "segment": seg,
    }


def _collect_ulgs(real_dir):
    pairs = []
    for f in sorted(real_dir.glob("*.ulg")):
        m = _TRIAL_RE.search(f.name)
        if m:
            pairs.append((int(m.group(1)), f))
    return sorted(pairs)


def _render_png(header, rows, out_path):
    """Render a table PNG with per-column widths sized to the widest cell."""
    plt = import_pyplot()

    padding_chars = 2
    col_char_widths = [
        max(len(str(col)), *(len(str(r[ci])) for r in rows)) + padding_chars
        for ci, col in enumerate(header)
    ]
    col_widths_in = [w * 0.10 for w in col_char_widths]
    total_w = sum(col_widths_in)
    fig_w = max(7.0, total_w + 0.6)
    fig_h = 0.55 + 0.55 * (len(rows) + 1)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    width_fracs = [w / total_w for w in col_widths_in]
    table = ax.table(cellText=rows, colLabels=header,
                     colWidths=width_fracs, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.7)  # vertical stretch only; widths come from colWidths

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.PAD = 0.12
        if r == 0:
            cell.set_facecolor("#e8ecf4")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#fafafa")

    ax.set_title("Table 6: Summary of seven real-world flight trials",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _fmt_time(x):
    return f"{round(x, 1):.1f}"


def main():
    p = argparse.ArgumentParser(description="Generate Table 6 (RQ6 real)")
    p.add_argument("--source", choices=["pre-baked", "fresh"], default="pre-baked")
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--png", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.results_dir:
        real_dir = Path(args.results_dir)
    else:
        real_dir = default_results_dir(args.source, "rq6") / "real_evaluation"

    if not real_dir.exists():
        print(f"No real_evaluation dir at {real_dir}", file=sys.stderr)
        return

    rows = []
    for trial, f in _collect_ulgs(real_dir):
        res = analyze_ulog(f, TARGET_LAT, TARGET_LON)
        if res is None:
            print(f"  trial{trial}: no attack data in {f.name}", file=sys.stderr)
            continue
        tm = TRIAL_METADATA.get(trial, {})
        rows.append({
            "trial": trial,
            "airframe": tm.get("airframe", "?"),
            "init_dist_m": res["init_dist_m"],
            "time_s": res["time_s"],
            "traj_length_m": res["traj_length_m"],
            "wind_mps": tm.get("wind_mps", ""),
            "gusts_mps": tm.get("gusts_mps", ""),
            "reached": res["reached"],
        })

    if not rows:
        print("No trials analyzed.", file=sys.stderr)
        return

    header = ["Trial", "Airframe", "Init. Dist. (m)", "Time (s)",
              "Traj. Len. (m)", "Wind (m/s)", "Gusts (m/s)"]
    cells = [[str(r["trial"]), r["airframe"],
              f"{r['init_dist_m']:.1f}",
              _fmt_time(r["time_s"]),
              f"{r['traj_length_m']:.1f}",
              r["wind_mps"], r["gusts_mps"]] for r in rows]

    emit_csv(header, cells, sourced_csv_path("rq6_table6", args.source))

    # Summary to stderr — keeps the CSV stdout clean for piping.
    avg_time = np.mean([r["time_s"] for r in rows])
    avg_len = np.mean([r["traj_length_m"] for r in rows])
    reached = sum(1 for r in rows if r["reached"])
    print(f"\n[rq6_table6] {reached}/{len(rows)} trials reached 10 m. "
          f"avg time={avg_time:.1f}s, avg traj_len={avg_len:.1f}m",
          file=sys.stderr)

    out_path = sourced_figure_path("rq6_table6", args.source, ".png")
    _render_png(header, cells, out_path)
    print(f"Table image saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
