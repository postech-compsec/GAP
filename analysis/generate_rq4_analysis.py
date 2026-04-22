#!/usr/bin/env python3
"""RQ4 CI-detector evasion analysis. Supports claim C4.

Usage:
    python3 -m analysis.generate_rq4_analysis --source pre-baked
    python3 -m analysis.generate_rq4_analysis --source pre-baked --trials 1,2,3

Reads:
    results/<source>/rq4/ by default

Writes:
    analysis/csv/rq4_analysis_<source>*.csv
    analysis/figures/rq4_analysis_<source>*.png

Notes:
    default --trials 1,2 matches the paper claim path
    --input-dir overrides the default input tree
"""

import argparse
import csv
import json
import re
from pathlib import Path

from analysis.common.table_utils import (
    csv_dir,
    figures_dir,
    project_root,
    save_grouped_bar_chart,
)

CRITERIA = ["cylinder_20m", "sphere_20m", "cylinder_10m", "sphere_10m"]
CRITERIA_DISPLAY = {
    "cylinder_20m": "Cyl. 20m",
    "sphere_20m": "Sph. 20m",
    "cylinder_10m": "Cyl. 10m",
    "sphere_10m": "Sph. 10m",
}

LOCATION_MAP = {
    1: ("North", 0.0),
    2: ("NNE", 30.0),
    3: ("ENE", 60.0),
    4: ("East", 90.0),
    5: ("ESE", 120.0),
    6: ("SSE", 150.0),
    7: ("South", 180.0),
    8: ("SSW", 210.0),
    9: ("WSW", 240.0),
    10: ("West", 270.0),
    11: ("WNW", 300.0),
    12: ("NNW", 330.0),
}

# Filename pattern: <ts>_gap-ci-detector_ardupilot-vm_trial<T>_w<NN>.json
_FN_RE = re.compile(
    r".*_gap-ci-detector_ardupilot-vm_trial(?P<trial>\d+)_w(?P<loc>\d+)\.json$"
)


def _project_root() -> Path:
    return project_root()


def _default_input_dir(source: str) -> Path:
    if source == "pre-baked":
        return _project_root() / "results" / "pre-baked" / "rq4"
    return _project_root() / "results" / "fresh" / "rq4"


def _default_output_dir() -> Path:
    return figures_dir()


def _canonical_location(location: int, direction, bearing_deg):
    canonical_direction, canonical_bearing = LOCATION_MAP.get(location, ("", None))
    return (
        direction if direction not in (None, "") else canonical_direction,
        bearing_deg if bearing_deg is not None else canonical_bearing,
    )


def _output_stem(source: str, allowed_trials: set[int]) -> str:
    default_trials = {1, 2}
    stem = f"rq4_analysis_{source}"
    if allowed_trials == {1, 2, 3}:
        return f"{stem}_all-trials"
    if allowed_trials != default_trials:
        ordered = "-".join(str(t) for t in sorted(allowed_trials))
        stem += f"_trials_{ordered}"
    return stem


def _load_records(input_dir: Path, allowed_trials: set[int]):
    records = []
    for p in sorted(input_dir.glob("*.json")):
        m = _FN_RE.match(p.name)
        if not m:
            continue
        trial = int(m.group("trial"))
        if trial not in allowed_trials:
            continue
        location = int(m.group("loc"))
        with p.open() as f:
            data = json.load(f)
        direction, bearing_deg = _canonical_location(
            location,
            data.get("direction", ""),
            data.get("bearing_deg"),
        )
        records.append({
            "path": p,
            "trial": trial,
            "location": location,
            "direction": direction,
            "bearing_deg": bearing_deg,
            "criteria_success": {c: bool(data.get(c, False)) for c in CRITERIA},
            "criteria_detected": {
                "sphere_20m": bool(data.get("detected_sph20", False)),
                "cylinder_20m": bool(data.get("detected_cyl20", False)),
                "sphere_10m": bool(data.get("detected_sph10", False)),
                "cylinder_10m": bool(data.get("detected_cyl10", False)),
            },
            "time_cyl10": data.get("time_cyl10"),
            "distance_cyl10": data.get("distance_cyl10"),
            "time_spent_s": data.get("time_spent_s"),
            "attack_steps": data.get("attack_steps"),
        })
    return records


def _summarize(records):
    """Return per-criterion {success_rate, ci_detect_rate_among_successes}."""
    n = len(records)
    summary = {}
    for c in CRITERIA:
        successes = [r for r in records if r["criteria_success"][c]]
        detected_within_success = [
            r for r in successes if r["criteria_detected"][c]
        ]
        summary[c] = {
            "n": n,
            "n_success": len(successes),
            "success_rate": (len(successes) / n) if n else 0.0,
            "n_ci_detected_of_success": len(detected_within_success),
            "ci_detect_rate_of_success": (
                len(detected_within_success) / len(successes)
                if successes else 0.0
            ),
        }
    return summary


def _write_csv(summary, out_csv: Path, records):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "criterion", "n_episodes", "n_success", "success_rate",
            "n_ci_detected_of_success", "ci_detect_rate_of_success",
        ])
        for c in CRITERIA:
            s = summary[c]
            w.writerow([
                c, s["n"], s["n_success"], f"{s['success_rate']:.3f}",
                s["n_ci_detected_of_success"],
                f"{s['ci_detect_rate_of_success']:.3f}",
            ])
        w.writerow([])
        w.writerow(["location", "trial", "direction", "bearing_deg", "cylinder_10m_success",
                    "cylinder_10m_ci_detected", "time_cyl10_s",
                    "distance_cyl10_m"])
        for r in sorted(records, key=lambda x: (x["location"], x["trial"])):
            w.writerow([
                r["location"], r["trial"], r["direction"],
                r["bearing_deg"],
                r["criteria_success"]["cylinder_10m"],
                r["criteria_detected"]["cylinder_10m"],
                r["time_cyl10"], r["distance_cyl10"],
            ])


def _write_plot(summary, out_png: Path, n_episodes: int):
    labels = [CRITERIA_DISPLAY[c] for c in CRITERIA]
    save_grouped_bar_chart(
        out_png,
        [
            ("Attack success rate (%)", [summary[c]["success_rate"] * 100 for c in CRITERIA]),
            (
                "CI-detector catch rate among successes (%)",
                [summary[c]["ci_detect_rate_of_success"] * 100 for c in CRITERIA],
            ),
        ],
        labels,
        f"RQ4 — CI-Detector Evasion (n={n_episodes})",
        ylabel="Percent",
        figsize=(10, 4.5),
        rotation=15,
        legend_kwargs={
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 0.995),
            "ncol": 2,
            "fontsize": 9,
            "framealpha": 0.9,
        },
        extra_headroom=14.0,
        tight_rect=(0, 0, 1, 0.97),
    )
    print(f"[saved PNG] {out_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the RQ4 CI-detector evasion analysis (C4, §7.4).",
    )
    parser.add_argument(
        "--source", choices=["pre-baked", "fresh"], default="pre-baked",
        help="Which result tree to analyze (default: pre-baked).",
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Override the default results/{source}/rq4/ directory.",
    )
    parser.add_argument(
        "--trials", default="1,2",
        help="Comma-separated trial numbers to include (default: paper-aligned trials 1,2).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write CSV + PNG (default: analysis/csv + analysis/figures).",
    )
    parser.add_argument("--png", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    allowed_trials = {int(t) for t in args.trials.split(",") if t.strip()}
    if not allowed_trials:
        raise SystemExit("[error] --trials empty")

    input_dir = Path(args.input_dir) if args.input_dir else _default_input_dir(args.source)
    if not input_dir.is_dir():
        raise SystemExit(f"[error] input dir not found: {input_dir}")
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    stem = _output_stem(args.source, allowed_trials)
    out_csv = (out_dir / f"{stem}.csv"
               if args.output_dir else csv_dir() / f"{stem}.csv")
    out_png = out_dir / f"{stem}.png"

    records = _load_records(input_dir, allowed_trials)
    if not records:
        raise SystemExit(
            f"[error] no JSONs matched trial filter {sorted(allowed_trials)} in {input_dir}"
        )

    summary = _summarize(records)

    print(f"\nRQ4 CI-detector evasion — source={args.source}, "
          f"trials={sorted(allowed_trials)}, n={len(records)} episodes")
    print("-" * 72)
    print(f"{'criterion':<14} {'n':>3} {'success':>12} "
          f"{'CI-caught/success':>18}")
    for c in CRITERIA:
        s = summary[c]
        print(f"{c:<14} {s['n']:>3} "
              f"{s['n_success']:>4}/{s['n']} ({s['success_rate']*100:5.1f}%)  "
              f"{s['n_ci_detected_of_success']:>4}/{s['n_success']} "
              f"({s['ci_detect_rate_of_success']*100:5.1f}%)")
    print("-" * 72)
    if allowed_trials == {1, 2}:
        print("Paper (§7.4 claim path): cylinder_10m success = 87.5%, "
              "CI-detected among successes = 23.8%.")
    else:
        print("Paper (§7.4 claim path) uses the default --trials 1,2. "
              "This run includes an explicit nondefault trial set.")
    print()

    _write_csv(summary, out_csv, records)
    print(f"[saved CSV] {out_csv}")
    _write_plot(summary, out_png, len(records))


if __name__ == "__main__":
    main()
