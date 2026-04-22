#!/usr/bin/env python3
"""RQ1 / Table 3 baseline comparison. Supports claim C1 (paper Table 3).

Usage:
    python3 -m analysis.generate_rq1_table3 --source pre-baked

Reads:
    results/<source>/rq1/

Writes:
    analysis/csv/rq1_table3_<source>.csv
    analysis/figures/rq1_table3_<source>.png
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

# Files follow: <ts>_<experiment>_<platform>[_<frame>][_<variant>][_w<NN>].json
# The experiment token is the second underscore-delimited chunk.
EXPERIMENT_DISPLAY = {
    "baseline-random":      "Random Bias",
    "baseline-directional": "Directional Bias",
    "baseline-adaptive":    "Adaptive Bias",
    "gap":                  "GAP (ours)",
}


def _experiment_of(filename: str) -> str:
    """Return the experiment token from a canonical filename (2nd chunk)."""
    parts = filename[:-len(".json")].split("_") if filename.endswith(".json") else filename.split("_")
    return parts[1] if len(parts) >= 2 else ""


def load_rates(results_dir: Path, experiment: str, source: str) -> dict:
    """Aggregate success-rate %% per criterion across all files for one experiment.

    For multi-worker runs (GAP) the per-worker counts are summed. For single-file
    runs (baselines) there's exactly one file.
    """
    files = [f for f in sorted(results_dir.glob("*.json")) if _experiment_of(f.name) == experiment]
    if not files:
        return {}

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
        return {}
    return {c: 100.0 * total_successes[c] / total_tests for c in CRITERIA}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate RQ1 Table 3")
    parser.add_argument("--source", choices=["pre-baked", "fresh"], default="fresh")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    parser.add_argument("--png", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = default_results_dir(args.source, "rq1")

    # Build rows in reporting order: baselines first, then GAP.
    rows = []
    for experiment in ("baseline-random", "baseline-directional", "baseline-adaptive", "gap"):
        rates = load_rates(results_dir, experiment, args.source)
        if rates:
            rows.append((EXPERIMENT_DISPLAY[experiment], rates))

    header = ["Method"] + CRITERIA_DISPLAY
    csv_rows = [
        [name] + [f"{rates.get(c, 0.0):.1f}" for c in CRITERIA]
        for name, rates in rows
    ]

    if not rows:
        print("No data found in", results_dir, file=sys.stderr)
        return

    emit_csv(header, csv_rows, sourced_csv_path("rq1_table3", args.source))
    out_path = sourced_figure_path("rq1_table3", args.source, ".png")
    save_grouped_bar_chart(
        out_path,
        [(name, [rates.get(c, 0.0) for c in CRITERIA]) for name, rates in rows],
        CRITERIA_DISPLAY,
        "Table 3: Baseline Comparison (RQ1)",
        figsize=(10, 5),
    )
    print(f"Chart saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
