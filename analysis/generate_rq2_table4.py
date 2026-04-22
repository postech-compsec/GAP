#!/usr/bin/env python3
"""RQ2 / Table 4 noise robustness. Supports claim C2 (paper Table 4).

Usage:
    python3 -m analysis.generate_rq2_table4 --source pre-baked

Reads:
    results/<source>/rq2/

Writes:
    analysis/csv/rq2_table4_<source>.csv
    analysis/figures/rq2_table4_<source>.png
"""

import argparse
import sys
from pathlib import Path

from analysis.common.table_utils import (
    default_results_dir,
    emit_csv,
    load_json,
    save_grouped_bar_chart,
    select_run_files,
    sourced_csv_path,
    sourced_figure_path,
    warn_bad_file,
)


CRITERIA = ["cylinder_20m", "sphere_20m", "cylinder_10m", "sphere_10m"]
CRITERIA_DISPLAY = ["Cyl. 20m", "Sph. 20m", "Cyl. 10m", "Sph. 10m"]

NOISE_CONDITIONS = [
    # (variant-tag-in-filename, display-name). None tag == clean run (no slot).
    (None,         "None"),
    ("tracking",   "Tracking Noise"),
    ("delay-loss", "Delay & Loss"),
    ("both",       "Both"),
]


def _variant_of(filename: str):
    """Extract the variant slot from a canonical filename, or None if absent.

    Format: <ts>_<experiment>_<platform>[_<frame>][_<variant>][_w<NN>].json
    For primary GAP runs there's no frame; the variant (if any) is the 4th
    underscore chunk unless that chunk begins with 'w' (worker index).
    """
    base = filename[:-len(".json")] if filename.endswith(".json") else filename
    parts = base.split("_")
    if len(parts) < 4:
        return None
    candidate = parts[3]
    if candidate.startswith("w") and candidate[1:].isdigit():
        return None  # no variant, that slot is the worker
    if candidate == "none":
        return None
    return candidate


def load_noise_results(results_dir, variant, source: str):
    """Aggregate per-criterion rates across every worker JSON for a variant."""
    files = [f for f in sorted(results_dir.glob("*_gap_px4-jmavsim_*.json"))
             if _variant_of(f.name) == variant]
    if not files:
        return None

    total_successes = {c: 0 for c in CRITERIA}
    total_tests = 0
    for f in select_run_files(files, source):
        try:
            data = load_json(f)
            summary = data["summary"]
            for c in CRITERIA:
                total_successes[c] += summary[c]["successes"]
            total_tests += summary["total_tests"]
        except Exception as exc:
            warn_bad_file(f, exc)
    if total_tests == 0:
        return None
    return {c: 100.0 * total_successes[c] / total_tests for c in CRITERIA}


def main():
    parser = argparse.ArgumentParser(description="Generate RQ2 Table 4")
    parser.add_argument("--source", choices=["pre-baked", "fresh"], default="fresh")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    parser.add_argument("--png", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = default_results_dir(args.source, "rq2")

    # Load per-condition data
    rows = []
    for tag, display in NOISE_CONDITIONS:
        rates = load_noise_results(results_dir, tag, args.source)
        if rates is not None:
            rows.append((display, rates))

    header = ["Noise Condition"] + CRITERIA_DISPLAY
    csv_rows = [
        [name] + [f"{rates.get(c, 0.0):.1f}" for c in CRITERIA]
        for name, rates in rows
    ]

    if not rows:
        print("No data found in", results_dir, file=sys.stderr)
        return

    emit_csv(header, csv_rows, sourced_csv_path("rq2_table4", args.source))
    out_path = sourced_figure_path("rq2_table4", args.source, ".png")
    save_grouped_bar_chart(
        out_path,
        [(name, [rates.get(c, 0.0) for c in CRITERIA]) for name, rates in rows],
        CRITERIA_DISPLAY,
        "Table 4: Noise Robustness (RQ2)",
        figsize=(10, 5),
    )
    print(f"Chart saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
