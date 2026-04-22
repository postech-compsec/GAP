#!/usr/bin/env python3
"""Check the paper metrics with pre-baked data or summarize fresh reruns.

Modes:
    pre-baked:
        Checks the paper metrics against the shipped pre-baked data. Reports
        PASS/SKIP under normal operation, and FAIL only on an internal
        mismatch in the shipped data.
    fresh:
        Supplementary observed-summary path against the same paper metrics.
        Reports OBSERVED/SKIP.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean

import numpy as np

from analysis.common.px4_attack_logs import prepare_px4_attack_csvs, summarize_px4_attack_csvs
from analysis.common.table_utils import default_results_dir, load_json, select_run_files
from analysis.generate_rq1_table3 import _experiment_of
from analysis.generate_rq2_table4 import _variant_of
from analysis.generate_rq3_table5 import _parse_filename
from analysis.generate_rq6_table6 import TARGET_LAT, TARGET_LON, _collect_ulgs, analyze_ulog
from analysis.generate_rq4_analysis import _load_records as load_rq4_records
from analysis.generate_rq4_analysis import _summarize as summarize_rq4


RQ1_EXPECTED = {
    "gap": ("C1", "GAP cyl-10 m success", 1021, 1200),
    "baseline-random": ("C1", "Random cyl-10 m success", 0, 24),
    "baseline-directional": ("C1", "Directional cyl-10 m success", 0, 24),
    "baseline-adaptive": ("C1", "Adaptive cyl-10 m success", 5, 24),
}

RQ2_EXPECTED = {
    "tracking": ("C2", "Tracking cyl-10 m success", 1088, 1220),
    "delay-loss": ("C2", "Delay/loss cyl-10 m success", 1020, 1168),
    "both": ("C2", "Both cyl-10 m success", 1061, 1218),
}

RQ3_EXPECTED = {
    ("px4-gazebo", "x500"): ("C3", "PX4 Gazebo x500 cyl-10 m success", 20, 24),
    ("ardupilot", "coaxcopter"): ("C3", "ArduPilot coaxcopter cyl-10 m success", 13, 24),
    ("ardupilot", "dodeca-hexa"): ("C3", "ArduPilot dodeca-hexa cyl-10 m success", 17, 24),
    ("ardupilot", "hexa"): ("C3", "ArduPilot hexa cyl-10 m success", 19, 24),
    ("ardupilot", "octa"): ("C3", "ArduPilot octa cyl-10 m success", 18, 24),
    ("ardupilot", "octaquad"): ("C3", "ArduPilot octaquad cyl-10 m success", 21, 24),
    ("ardupilot", "quad"): ("C3", "ArduPilot quad cyl-10 m success", 19, 24),
    ("ardupilot", "singlecopter"): ("C3", "ArduPilot singlecopter cyl-10 m success", 20, 24),
    ("ardupilot", "tri"): ("C3", "ArduPilot tri cyl-10 m success", 19, 24),
    ("ardupilot", "y6"): ("C3", "ArduPilot y6 cyl-10 m success", 19, 24),
}

RQ5_EXPECTED = {
    "successful_attack_episodes": 1048,
    "gyro_failsafe_triggered_count": 0,
    "other_failsafe_triggered_count": 213,
    "avg_deactivation_time_s": 2.121,
}

RQ4_EXPECTED = {
    "n_success": 21,
    "n_total": 24,
    "n_ci_detected_of_success": 5,
    "n_success_for_ci": 21,
}

RQ6_EXPECTED = {
    "reached": 7,
    "total": 7,
    "avg_time_s": 13.794,
    "avg_path_length_m": 53.576,
}

@dataclass
class ClaimResult:
    claim: str
    metric: str
    source: str
    observed: str
    expected: str
    difference: str
    status: str


def _skip(claim: str, metric: str, source: str, expected: str, note: str = "missing") -> ClaimResult:
    return ClaimResult(claim, metric, source, note, expected, "n/a", "SKIP")


def _fmt_ratio(successes: int, total: int) -> str:
    pct = 100.0 * successes / total if total else 0.0
    return f"{successes}/{total} ({pct:.3f}%)"


def _fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _ratio_pct(successes: int, total: int) -> float:
    return 100.0 * successes / total if total else 0.0


def _fmt_pp_delta(
    observed_successes: int,
    observed_total: int,
    expected_successes: int,
    expected_total: int,
) -> str:
    delta = _ratio_pct(observed_successes, observed_total) - _ratio_pct(expected_successes, expected_total)
    return f"{delta:+.3f} pp"


def _fmt_scalar_delta(observed: float, expected: float, unit: str = "", digits: int = 3) -> str:
    suffix = f" {unit}" if unit else ""
    return f"{observed - expected:+.{digits}f}{suffix}"


def _ratio_status(
    observed_successes: int,
    observed_total: int,
    expected_successes: int,
    expected_total: int,
    direction: str,
) -> str:
    observed_pct = _ratio_pct(observed_successes, observed_total)
    expected_pct = _ratio_pct(expected_successes, expected_total)
    if direction == "at_least":
        return "MET" if observed_pct + 1e-9 >= expected_pct else "UNMET"
    if direction == "at_most":
        return "MET" if observed_pct <= expected_pct + 1e-9 else "UNMET"
    raise ValueError(f"unsupported direction: {direction}")


def _scalar_status(observed: float, expected: float, direction: str) -> str:
    if direction == "at_least":
        return "MET" if observed + 1e-9 >= expected else "UNMET"
    if direction == "at_most":
        return "MET" if observed <= expected + 1e-9 else "UNMET"
    raise ValueError(f"unsupported direction: {direction}")


def _reference_ratio_status(
    observed_successes: int,
    observed_total: int,
    expected_successes: int,
    expected_total: int,
    direction: str,
) -> str:
    return "PASS" if _ratio_status(
        observed_successes,
        observed_total,
        expected_successes,
        expected_total,
        direction,
    ) == "MET" else "FAIL"


def _reference_scalar_status(
    observed: float,
    expected: float,
    direction: str,
    *,
    digits: int | None = None,
) -> str:
    if digits is not None:
        observed = round(observed, digits)
        expected = round(expected, digits)
    return "PASS" if _scalar_status(observed, expected, direction) == "MET" else "FAIL"


def _observed_row(
    claim: str,
    metric: str,
    source: str,
    observed: str,
    expected: str,
    difference: str = "n/a",
) -> ClaimResult:
    return ClaimResult(claim, metric, source, observed, expected, difference, "OBSERVED")


def _aggregate_cyl10_counts(files) -> tuple[int, int]:
    total_successes = 0
    total_tests = 0
    for path in files:
        data = load_json(path)
        summary = data["summary"]
        total_successes += summary["cylinder_10m"]["successes"]
        total_tests += summary["total_tests"]
    return total_successes, total_tests


def _load_rq1_counts(results_dir, experiment: str, source: str) -> tuple[int, int] | None:
    files = [f for f in sorted(results_dir.glob("*.json")) if _experiment_of(f.name) == experiment]
    if not files:
        return None
    return _aggregate_cyl10_counts(select_run_files(files, source))


def _load_rq2_counts(results_dir, variant: str, source: str) -> tuple[int, int] | None:
    files = [
        f
        for f in sorted(results_dir.glob("*_gap_px4-jmavsim_*.json"))
        if _variant_of(f.name) == variant
    ]
    if not files:
        return None
    return _aggregate_cyl10_counts(select_run_files(files, source))


def _load_rq3_counts(results_dir, source: str) -> dict[tuple[str, str], tuple[int, int]]:
    grouped = {}
    for path in sorted(results_dir.glob("*_gap_*.json")):
        platform, frame = _parse_filename(path.name)
        if platform is None:
            continue
        grouped.setdefault((platform, frame), []).append(path)

    out = {}
    for key, files in grouped.items():
        total_successes = 0
        total_tests = 0
        for path in select_run_files(files, source):
            data = load_json(path)
            summary = data["summary"]
            tests = summary.get("total_tests", 0)
            if tests == 0:
                continue
            total_successes += summary["cylinder_10m"]["successes"]
            total_tests += tests
        if total_tests > 0:
            out[key] = (total_successes, total_tests)
    return out


def _load_rq3_runs(results_dir, source: str) -> dict[tuple[str, str], list[tuple[str, int, int]]]:
    grouped = {}
    for path in sorted(results_dir.glob("*_gap_*.json")):
        platform, frame = _parse_filename(path.name)
        if platform is None:
            continue
        grouped.setdefault((platform, frame), []).append(path)

    out = {}
    for key, files in grouped.items():
        runs = []
        for path in select_run_files(files, source):
            data = load_json(path)
            summary = data["summary"]
            tests = summary.get("total_tests", 0)
            if tests == 0:
                continue
            runs.append((path.name, summary["cylinder_10m"]["successes"], tests))
        if runs:
            out[key] = runs
    return out


def _fmt_rq3_fresh_summary(
    runs: list[tuple[str, int, int]],
    expected_successes: int,
    expected_total: int,
) -> tuple[str, str]:
    successes = [succ for _name, succ, _tests in runs]
    tests = [tests for _name, _succ, tests in runs]
    mean_pct = mean(_ratio_pct(succ, tests) for _name, succ, tests in runs)
    difference = f"{mean_pct - _ratio_pct(expected_successes, expected_total):+.3f} pp"
    if len(set(tests)) == 1:
        total = tests[0]
        obs = (
            f"mean {mean(successes):.1f}/{total}, "
            f"range {min(successes)}..{max(successes)}/{total} "
            f"({len(successes)} runs)"
        )
        return obs, difference

    obs = (
        f"mean {mean(successes):.1f} successes, "
        f"range {min(successes)}..{max(successes)} "
        f"({len(successes)} runs)"
    )
    return obs, difference


def _load_rq5_metrics(source: str) -> dict | None:
    paths = prepare_px4_attack_csvs(source, verbose=False)
    if not paths:
        return None
    return summarize_px4_attack_csvs(paths)


def verify_rq1(source: str) -> list[ClaimResult]:
    results_dir = default_results_dir(source, "rq1")
    observed = {experiment: _load_rq1_counts(results_dir, experiment, source) for experiment in RQ1_EXPECTED}

    rows = []
    for experiment, (claim, metric, exp_x, exp_n) in RQ1_EXPECTED.items():
        if observed[experiment] is None:
            rows.append(_skip(claim, metric, source, _fmt_ratio(exp_x, exp_n)))
            continue
        obs_x, obs_n = observed[experiment]
        if source == "pre-baked":
            rows.append(ClaimResult(
                claim=claim,
                metric=metric,
                source=source,
                observed=_fmt_ratio(obs_x, obs_n),
                expected=_fmt_ratio(exp_x, exp_n),
                difference=_fmt_pp_delta(obs_x, obs_n, exp_x, exp_n),
                status=_reference_ratio_status(obs_x, obs_n, exp_x, exp_n, "at_least"),
            ))
        else:
            rows.append(_observed_row(
                claim=claim,
                metric=metric,
                source=source,
                observed=_fmt_ratio(obs_x, obs_n),
                expected=_fmt_ratio(exp_x, exp_n),
                difference=_fmt_pp_delta(obs_x, obs_n, exp_x, exp_n),
            ))
    return rows


def verify_rq2(source: str) -> list[ClaimResult]:
    results_dir = default_results_dir(source, "rq2")
    observed = {variant: _load_rq2_counts(results_dir, variant, source) for variant in RQ2_EXPECTED}

    rows = []
    for variant, (claim, metric, exp_x, exp_n) in RQ2_EXPECTED.items():
        if observed[variant] is None:
            rows.append(_skip(claim, metric, source, _fmt_ratio(exp_x, exp_n)))
            continue
        obs_x, obs_n = observed[variant]
        if source == "pre-baked":
            rows.append(ClaimResult(
                claim=claim,
                metric=metric,
                source=source,
                observed=_fmt_ratio(obs_x, obs_n),
                expected=_fmt_ratio(exp_x, exp_n),
                difference=_fmt_pp_delta(obs_x, obs_n, exp_x, exp_n),
                status=_reference_ratio_status(obs_x, obs_n, exp_x, exp_n, "at_least"),
            ))
        else:
            rows.append(_observed_row(
                claim=claim,
                metric=metric,
                source=source,
                observed=_fmt_ratio(obs_x, obs_n),
                expected=_fmt_ratio(exp_x, exp_n),
                difference=_fmt_pp_delta(obs_x, obs_n, exp_x, exp_n),
            ))
    return rows


def verify_rq3(source: str) -> list[ClaimResult]:
    results_dir = default_results_dir(source, "rq3")
    if source == "pre-baked":
        results = _load_rq3_counts(results_dir, source)
    else:
        results = _load_rq3_runs(results_dir, source)
    rows = []
    for key, (claim, metric, exp_x, exp_n) in RQ3_EXPECTED.items():
        if key not in results:
            rows.append(_skip(
                claim,
                metric,
                source,
                _fmt_ratio(exp_x, exp_n),
            ))
            continue
        if source == "pre-baked":
            obs_x, obs_n = results[key]
            rows.append(ClaimResult(
                claim=claim,
                metric=metric,
                source=source,
                observed=_fmt_ratio(obs_x, obs_n),
                expected=_fmt_ratio(exp_x, exp_n),
                difference=_fmt_pp_delta(obs_x, obs_n, exp_x, exp_n),
                status=_reference_ratio_status(obs_x, obs_n, exp_x, exp_n, "at_least"),
            ))
        else:
            observed, difference = _fmt_rq3_fresh_summary(results[key], exp_x, exp_n)
            rows.append(_observed_row(
                claim=claim,
                metric=metric,
                source=source,
                observed=observed,
                expected=_fmt_ratio(exp_x, exp_n),
                difference=difference,
            ))
    return rows


def verify_rq5(source: str) -> list[ClaimResult]:
    summary = _load_rq5_metrics(source)
    if summary is None:
        return [
            _skip("C5", "Gyro failsafe triggers", source, "0"),
            _skip("C5", "Other failsafe rate", source, _fmt_ratio(213, 1048)),
            _skip("C5", "Avg deactivation time (s)", source, "2.121"),
        ]

    if source == "fresh" and summary["successful_attack_episodes"] == 0:
        return [
            _skip("C5", "Gyro failsafe triggers", source, "0", note="no successful episodes"),
            _skip("C5", "Other failsafe rate", source, _fmt_ratio(213, 1048), note="no successful episodes"),
            _skip("C5", "Avg deactivation time (s)", source, "2.121", note="no successful episodes"),
        ]

    gyro_failsafe_count = summary["gyro_failsafe_triggered_count"]
    observed_time = summary["avg_deactivation_time_s"]
    observed_time_str = "N/A" if observed_time is None else _fmt_float(observed_time)
    observed_other_count = summary["other_failsafe_triggered_count"]
    observed_success_count = summary["successful_attack_episodes"]
    if source == "pre-baked":
        return [
            ClaimResult(
                "C5",
                "Gyro failsafe triggers",
                source,
                str(gyro_failsafe_count),
                "0",
                _fmt_scalar_delta(
                    float(gyro_failsafe_count),
                    float(RQ5_EXPECTED["gyro_failsafe_triggered_count"]),
                    digits=0,
                ),
                _reference_scalar_status(
                    float(gyro_failsafe_count),
                    float(RQ5_EXPECTED["gyro_failsafe_triggered_count"]),
                    "at_most",
                    digits=0,
                ),
            ),
            ClaimResult(
                "C5",
                "Other failsafe rate",
                source,
                _fmt_ratio(observed_other_count, observed_success_count),
                _fmt_ratio(
                    RQ5_EXPECTED["other_failsafe_triggered_count"],
                    RQ5_EXPECTED["successful_attack_episodes"],
                ),
                _fmt_pp_delta(
                    observed_other_count,
                    observed_success_count,
                    RQ5_EXPECTED["other_failsafe_triggered_count"],
                    RQ5_EXPECTED["successful_attack_episodes"],
                ),
                _reference_ratio_status(
                    observed_other_count,
                    observed_success_count,
                    RQ5_EXPECTED["other_failsafe_triggered_count"],
                    RQ5_EXPECTED["successful_attack_episodes"],
                    "at_most",
                ),
            ),
            ClaimResult(
                "C5",
                "Avg deactivation time (s)",
                source,
                observed_time_str,
                _fmt_float(RQ5_EXPECTED["avg_deactivation_time_s"]),
                "n/a" if observed_time is None else _fmt_scalar_delta(
                    observed_time,
                    RQ5_EXPECTED["avg_deactivation_time_s"],
                    unit="s",
                ),
                "FAIL" if observed_time is None else _reference_scalar_status(
                    observed_time,
                    RQ5_EXPECTED["avg_deactivation_time_s"],
                    "at_most",
                    digits=3,
                ),
            ),
        ]

    return [
        _observed_row(
            "C5",
            "Gyro failsafe triggers",
            source,
            str(gyro_failsafe_count),
            "0",
            _fmt_scalar_delta(
                float(gyro_failsafe_count),
                float(RQ5_EXPECTED["gyro_failsafe_triggered_count"]),
                digits=0,
            ),
        ),
        _observed_row(
            "C5",
            "Other failsafe rate",
            source,
            _fmt_ratio(observed_other_count, observed_success_count),
            _fmt_ratio(
                RQ5_EXPECTED["other_failsafe_triggered_count"],
                RQ5_EXPECTED["successful_attack_episodes"],
            ),
            _fmt_pp_delta(
                observed_other_count,
                observed_success_count,
                RQ5_EXPECTED["other_failsafe_triggered_count"],
                RQ5_EXPECTED["successful_attack_episodes"],
            ),
        ),
        _observed_row(
            "C5",
            "Avg deactivation time (s)",
            source,
            observed_time_str,
            _fmt_float(RQ5_EXPECTED["avg_deactivation_time_s"]),
            "n/a" if observed_time is None else _fmt_scalar_delta(
                observed_time,
                RQ5_EXPECTED["avg_deactivation_time_s"],
                unit="s",
            ),
        ),
    ]


def verify_rq4(source: str) -> list[ClaimResult]:
    results_dir = default_results_dir(source, "rq4")
    if not results_dir.exists() or not list(results_dir.glob("*.json")):
        return [
            _skip("C4", "RQ4 cyl-10 m success", source, _fmt_ratio(21, 24)),
            _skip("C4", "RQ4 CI-caught among successes", source, _fmt_ratio(5, 21)),
        ]
    summary = summarize_rq4(load_rq4_records(results_dir, {1, 2}))
    if source == "pre-baked":
        return [
            ClaimResult(
                "C4",
                "RQ4 cyl-10 m success",
                source,
                _fmt_ratio(summary["cylinder_10m"]["n_success"], summary["cylinder_10m"]["n"]),
                _fmt_ratio(RQ4_EXPECTED["n_success"], RQ4_EXPECTED["n_total"]),
                _fmt_pp_delta(
                    summary["cylinder_10m"]["n_success"],
                    summary["cylinder_10m"]["n"],
                    RQ4_EXPECTED["n_success"],
                    RQ4_EXPECTED["n_total"],
                ),
                _reference_ratio_status(
                    summary["cylinder_10m"]["n_success"],
                    summary["cylinder_10m"]["n"],
                    RQ4_EXPECTED["n_success"],
                    RQ4_EXPECTED["n_total"],
                    "at_least",
                ),
            ),
            ClaimResult(
                "C4",
                "RQ4 CI-caught among successes",
                source,
                _fmt_ratio(
                    summary["cylinder_10m"]["n_ci_detected_of_success"],
                    summary["cylinder_10m"]["n_success"],
                ),
                _fmt_ratio(
                    RQ4_EXPECTED["n_ci_detected_of_success"],
                    RQ4_EXPECTED["n_success_for_ci"],
                ),
                _fmt_pp_delta(
                    summary["cylinder_10m"]["n_ci_detected_of_success"],
                    summary["cylinder_10m"]["n_success"],
                    RQ4_EXPECTED["n_ci_detected_of_success"],
                    RQ4_EXPECTED["n_success_for_ci"],
                ),
                _reference_ratio_status(
                    summary["cylinder_10m"]["n_ci_detected_of_success"],
                    summary["cylinder_10m"]["n_success"],
                    RQ4_EXPECTED["n_ci_detected_of_success"],
                    RQ4_EXPECTED["n_success_for_ci"],
                    "at_most",
                ),
            ),
        ]

    return [
        _observed_row(
            "C4",
            "RQ4 cyl-10 m success",
            source,
            _fmt_ratio(summary["cylinder_10m"]["n_success"], summary["cylinder_10m"]["n"]),
            _fmt_ratio(RQ4_EXPECTED["n_success"], RQ4_EXPECTED["n_total"]),
            _fmt_pp_delta(
                summary["cylinder_10m"]["n_success"],
                summary["cylinder_10m"]["n"],
                RQ4_EXPECTED["n_success"],
                RQ4_EXPECTED["n_total"],
            ),
        ),
        _observed_row(
            "C4",
            "RQ4 CI-caught among successes",
            source,
            _fmt_ratio(
                summary["cylinder_10m"]["n_ci_detected_of_success"],
                summary["cylinder_10m"]["n_success"],
            ),
            _fmt_ratio(
                RQ4_EXPECTED["n_ci_detected_of_success"],
                RQ4_EXPECTED["n_success_for_ci"],
            ),
            _fmt_pp_delta(
                summary["cylinder_10m"]["n_ci_detected_of_success"],
                summary["cylinder_10m"]["n_success"],
                RQ4_EXPECTED["n_ci_detected_of_success"],
                RQ4_EXPECTED["n_success_for_ci"],
            ),
        ),
    ]


def verify_rq6(source: str) -> list[ClaimResult]:
    real_dir = default_results_dir(source, "rq6") / "real_evaluation"
    if not real_dir.exists() or not list(real_dir.glob("*.ulg")):
        return [
            _skip("C6", "Real-flight successes", source, _fmt_ratio(7, 7)),
            _skip("C6", "Avg time to 10 m cylinder (s)", source, "13.794"),
            _skip("C6", "Avg path length (m)", source, "53.576"),
        ]

    rows = []
    for _trial, path in _collect_ulgs(real_dir):
        result = analyze_ulog(path, TARGET_LAT, TARGET_LON)
        if result is not None:
            rows.append(result)
    if not rows:
        return [
            _skip("C6", "Real-flight successes", source, _fmt_ratio(7, 7)),
            _skip("C6", "Avg time to 10 m cylinder (s)", source, "13.794"),
            _skip("C6", "Avg path length (m)", source, "53.576"),
        ]

    reached = sum(1 for row in rows if row["reached"])
    avg_time = float(np.mean([row["time_s"] for row in rows]))
    avg_len = float(np.mean([row["traj_length_m"] for row in rows]))

    if source == "pre-baked":
        return [
            ClaimResult(
                "C6",
                "Real-flight successes",
                source,
                _fmt_ratio(reached, len(rows)),
                _fmt_ratio(RQ6_EXPECTED["reached"], RQ6_EXPECTED["total"]),
                _fmt_pp_delta(reached, len(rows), RQ6_EXPECTED["reached"], RQ6_EXPECTED["total"]),
                _reference_ratio_status(reached, len(rows), RQ6_EXPECTED["reached"], RQ6_EXPECTED["total"], "at_least"),
            ),
            ClaimResult(
                "C6",
                "Avg time to 10 m cylinder (s)",
                source,
                _fmt_float(avg_time),
                _fmt_float(RQ6_EXPECTED["avg_time_s"]),
                _fmt_scalar_delta(avg_time, RQ6_EXPECTED["avg_time_s"], unit="s"),
                _reference_scalar_status(avg_time, RQ6_EXPECTED["avg_time_s"], "at_most", digits=3),
            ),
            ClaimResult(
                "C6",
                "Avg path length (m)",
                source,
                _fmt_float(avg_len),
                _fmt_float(RQ6_EXPECTED["avg_path_length_m"]),
                _fmt_scalar_delta(avg_len, RQ6_EXPECTED["avg_path_length_m"], unit="m"),
                _reference_scalar_status(avg_len, RQ6_EXPECTED["avg_path_length_m"], "at_most", digits=3),
            ),
        ]

    return [
        _observed_row(
            "C6",
            "Real-flight successes",
            source,
            _fmt_ratio(reached, len(rows)),
            _fmt_ratio(RQ6_EXPECTED["reached"], RQ6_EXPECTED["total"]),
            _fmt_pp_delta(reached, len(rows), RQ6_EXPECTED["reached"], RQ6_EXPECTED["total"]),
        ),
        _observed_row(
            "C6",
            "Avg time to 10 m cylinder (s)",
            source,
            _fmt_float(avg_time),
            _fmt_float(RQ6_EXPECTED["avg_time_s"]),
            _fmt_scalar_delta(avg_time, RQ6_EXPECTED["avg_time_s"], unit="s"),
        ),
        _observed_row(
            "C6",
            "Avg path length (m)",
            source,
            _fmt_float(avg_len),
            _fmt_float(RQ6_EXPECTED["avg_path_length_m"]),
            _fmt_scalar_delta(avg_len, RQ6_EXPECTED["avg_path_length_m"], unit="m"),
        ),
    ]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check the paper metrics with pre-baked data or summarize fresh "
            "reruns against those same metrics."
        )
    )
    parser.add_argument("source_pos", nargs="?", choices=["pre-baked", "fresh"])
    parser.add_argument("--source", dest="source_flag", choices=["pre-baked", "fresh"])
    args = parser.parse_args()
    source = args.source_flag or args.source_pos or "pre-baked"

    if source == "pre-baked":
        print("Mode: checking paper metrics with pre-baked data.")
        print("Status: PASS, SKIP, or FAIL on shipped-data mismatch.")
    else:
        print("Mode: summarizing fresh reruns.")
        print("Status: OBSERVED or SKIP. No PASS/SKIP in fresh mode.")
    print()

    rows = (
        verify_rq1(source)
        + verify_rq2(source)
        + verify_rq3(source)
        + verify_rq4(source)
        + verify_rq5(source)
        + verify_rq6(source)
    )

    claim_hdr = "Claim"
    metric_hdr = "Metric"
    src_hdr = "Src"
    obs_hdr = "Observed"
    exp_hdr = "Expected"
    diff_hdr = "Delta"

    claim_w = max(len(claim_hdr), *(len(row.claim) for row in rows))
    metric_w = max(len(metric_hdr), *(len(row.metric) for row in rows))
    src_w = max(len(src_hdr), *(len(row.source) for row in rows))
    obs_w = max(len(obs_hdr), *(len(row.observed) for row in rows))
    exp_w = max(len(exp_hdr), *(len(row.expected) for row in rows))
    diff_w = max(len(diff_hdr), *(len(row.difference) for row in rows))

    header = (
        f"{claim_hdr:<{claim_w}}  {metric_hdr:<{metric_w}}  {src_hdr:<{src_w}}  "
        f"{obs_hdr:>{obs_w}}  {exp_hdr:>{exp_w}}  {diff_hdr:>{diff_w}}  Status"
    )
    print("Delta = observed - expected.")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.claim:<{claim_w}}  {row.metric:<{metric_w}}  {row.source:<{src_w}}  "
            f"{row.observed:>{obs_w}}  {row.expected:>{exp_w}}  {row.difference:>{diff_w}}  {row.status}"
        )


if __name__ == "__main__":
    main()
