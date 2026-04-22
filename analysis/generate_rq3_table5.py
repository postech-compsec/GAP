#!/usr/bin/env python3
"""RQ3 / Table 5 cross-platform transfer. Supports claim C3 (paper Table 5).

Usage:
    python3 -m analysis.generate_rq3_table5 --source pre-baked

Reads:
    results/<source>/rq3/
    filenames of the form <ts>_gap_<platform>_<frame>.json

Writes:
    analysis/csv/rq3_table5_<source>.csv
    analysis/figures/rq3_table5_<source>.png
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


def _parse_filename(filename: str):
    """Return (platform, frame) from a canonical GAP cross-platform filename.

    Expected: ``<ts>_gap_<platform>_<frame>.json`` — where platform can itself
    contain hyphens (e.g. ``px4-gazebo``) but no underscores.
    """
    base = filename[:-len(".json")] if filename.endswith(".json") else filename
    parts = base.split("_")
    if len(parts) != 4 or parts[1] != "gap":
        return None, None
    return parts[2], parts[3]


def load_results(results_dir: Path, source: str):
    """Aggregate {(platform, frame): {criterion: rate_pct}} from selected files."""
    grouped = {}
    for f in sorted(results_dir.glob("*_gap_*.json")):
        platform, frame = _parse_filename(f.name)
        if platform is None:
            continue
        grouped.setdefault((platform, frame), []).append(f)

    out = {}
    for key, files in grouped.items():
        total_successes = {c: 0 for c in CRITERIA}
        total_tests = 0
        for f in select_run_files(files, source):
            try:
                data = load_json(f)
                summary = data["summary"]
                tests = summary.get("total_tests", 0)
                if tests == 0:
                    continue
                for c in CRITERIA:
                    total_successes[c] += summary[c]["successes"]
                total_tests += tests
            except Exception as exc:
                warn_bad_file(f, exc)
        if total_tests > 0:
            out[key] = {c: 100.0 * total_successes[c] / total_tests for c in CRITERIA}
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate RQ3 Table 5")
    parser.add_argument("--source", choices=["pre-baked", "fresh"], default="fresh")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    parser.add_argument("--png", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = default_results_dir(args.source, "rq3")

    results = load_results(results_dir, args.source)
    sorted_keys = sorted(results.keys())

    header = ["Platform", "Frame"] + CRITERIA_DISPLAY
    csv_rows = []
    for platform, frame in sorted_keys:
        rates = results[(platform, frame)]
        csv_rows.append([platform, frame] + [f"{rates.get(c, 0.0):.1f}" for c in CRITERIA])

    if not results:
        print("No data found in", results_dir, file=sys.stderr)
        return

    emit_csv(header, csv_rows, sourced_csv_path("rq3_table5", args.source))
    out_path = sourced_figure_path("rq3_table5", args.source, ".png")
    save_grouped_bar_chart(
        out_path,
        [
            (f"{platform}/{frame}", [results[(platform, frame)].get(c, 0.0) for c in CRITERIA])
            for platform, frame in sorted_keys
        ],
        CRITERIA_DISPLAY,
        "Table 5: Cross-Platform Transfer (RQ3)",
        figsize=(max(12, len(sorted_keys) * 1.5), 6),
        legend_kwargs={"bbox_to_anchor": (1.02, 1), "loc": "upper left", "fontsize": 8},
        value_fontsize=6,
    )
    print(f"Chart saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
