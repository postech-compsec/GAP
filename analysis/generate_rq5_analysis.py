#!/usr/bin/env python3
"""RQ5 failsafe analysis over the shared PX4 attack-flight corpus. Supports claim C5.

Usage:
    python3 -m analysis.generate_rq5_analysis --source pre-baked
    python3 -m analysis.generate_rq5_analysis --source pre-baked --use-ulogs

Reads:
    pre-baked: results/<source>/flight-logs/px4/csv/ by default
    fresh: results/<source>/flight-logs/px4/raw/ first, then cached CSVs

Writes:
    analysis/csv/rq5_analysis_<source>.csv

Notes:
    --use-ulogs forces a CSV refresh from results/<source>/flight-logs/px4/raw/
"""

from __future__ import annotations

import argparse
import sys

from analysis.common.px4_attack_logs import (
    prepare_px4_attack_csvs,
    px4_attack_csv_root,
    summarize_px4_attack_csvs,
)
from analysis.common.table_utils import emit_csv, sourced_csv_path


def _fmt_optional(value, digits: int) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="Generate RQ5 failsafe analysis")
    parser.add_argument("--source", choices=["pre-baked", "fresh"], default="pre-baked")
    parser.add_argument(
        "--use-ulogs",
        action="store_true",
        help="Force a worker-CSV refresh from the raw ULogs first.",
    )
    args = parser.parse_args()

    paths = prepare_px4_attack_csvs(
        args.source,
        force_refresh=args.use_ulogs,
        verbose=True,
    )
    if not paths:
        print(f"No RQ5 worker CSVs found under {px4_attack_csv_root(args.source)}", file=sys.stderr)
        return

    summary = summarize_px4_attack_csvs(paths)
    header = ["Metric", "Value"]
    rows = [
        ["Total Flight Logs", str(summary["total_ulogs"])],
        ["Successful Attack Episodes", str(summary["successful_attack_episodes"])],
        ["Gyro Failsafe Triggered Count", str(summary["gyro_failsafe_triggered_count"])],
        ["Gyro Failsafe Triggered (%)", f'{summary["gyro_failsafe_triggered_pct"]:.2f}'],
        ["Other Failsafe Triggered Count", str(summary["other_failsafe_triggered_count"])],
        ["Other Failsafe Triggered (%)", f'{summary["other_failsafe_triggered_pct"]:.2f}'],
        ["Avg Deactivation Time (s)", _fmt_optional(summary["avg_deactivation_time_s"], 3)],
        ["Max Deactivation Time (s)", _fmt_optional(summary["max_deactivation_time_s"], 3)],
    ]
    emit_csv(header, rows, sourced_csv_path("rq5_analysis", args.source))
    print(
        f"[rq5_analysis] {summary['other_failsafe_triggered_count']}/"
        f"{summary['successful_attack_episodes']} other-failsafe triggers, "
        f"avg deactivation={_fmt_optional(summary['avg_deactivation_time_s'], 3)}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
