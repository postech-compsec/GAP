#!/usr/bin/env python3
"""Small shared helpers for analysis table scripts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_results_dir(source: str, rq: str) -> Path:
    return project_root() / "results" / source / rq


def figures_dir() -> Path:
    out_dir = project_root() / "analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def csv_dir() -> Path:
    out_dir = project_root() / "analysis" / "csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def sourced_stem(stem: str, source: str) -> str:
    return f"{stem}_{source}"


def sourced_csv_path(stem: str, source: str) -> Path:
    return csv_dir() / f"{sourced_stem(stem, source)}.csv"


def sourced_figure_path(stem: str, source: str, suffix: str) -> Path:
    return figures_dir() / f"{sourced_stem(stem, source)}{suffix}"


def load_json(path: Path):
    return json.loads(path.read_text())


def timestamp_of(filename: str) -> str:
    base = filename[:-len(".json")] if filename.endswith(".json") else filename
    return base.split("_")[0] if base else ""


def run_group_of(filename: str) -> str:
    """Keep adjacent-second worker batches together as one logical run."""
    ts = timestamp_of(filename)
    return ts[:-2] if len(ts) >= 2 else ts


def latest_run_files(files):
    files = list(files)
    if not files:
        return []
    latest_group = max(run_group_of(f.name) for f in files)
    return [f for f in files if run_group_of(f.name) == latest_group]


def select_run_files(files, source: str):
    files = list(files)
    if source == "fresh":
        return files
    return latest_run_files(files)


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def emit_csv(header, rows, out_csv: Path):
    write_csv(out_csv, header, rows)
    writer = csv.writer(sys.stdout)
    writer.writerow(header)
    writer.writerows(rows)


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def warn_bad_file(path: Path, exc: Exception):
    print(f"Warning: {path.name}: {exc}", file=sys.stderr)


def save_grouped_bar_chart(
    out_path: Path,
    series,
    category_labels,
    title: str,
    ylabel: str = "Success Rate (%)",
    figsize=(10, 5),
    rotation: int = 0,
    legend_kwargs: dict | None = None,
    value_fontsize: int = 8,
    extra_headroom: float = 0.0,
    tight_rect=(0, 0, 1, 0.97),
    x_extra=(0.0, 0.0),
):
    """Save a grouped bar chart with per-bar percentage labels."""
    import numpy as np

    plt = import_pyplot()
    series = [(name, list(values)) for name, values in series]
    x = np.arange(len(category_labels))
    width = 0.8 / max(len(series), 1)
    max_value = max((max(values) for _, values in series), default=0.0)
    headroom = max(6.0, 0.08 * max(100.0, max_value) + 1.0)
    label_offset = max(0.8, headroom * 0.22)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, values) in enumerate(series):
        bars = ax.bar(x + i * width, values, width, label=name, linewidth=1.0)
        for bar, value in zip(bars, values):
            if value == 0:
                ax.hlines(
                    -0.8,
                    bar.get_x(),
                    bar.get_x() + bar.get_width(),
                    colors=[bar.get_facecolor()],
                    linewidth=4,
                    zorder=bar.get_zorder() + 1,
                )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + label_offset,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                clip_on=False,
            )

    ax.set_ylabel(ylabel)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(category_labels, rotation=rotation)
    ax.set_ylim(-2.0, max(105.0, max_value + headroom + extra_headroom))
    if x_extra != (0.0, 0.0):
        ax.set_xlim(-0.5 - x_extra[0], len(category_labels) - 0.5 + x_extra[1])
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(**(legend_kwargs or {}))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=tight_rect)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
