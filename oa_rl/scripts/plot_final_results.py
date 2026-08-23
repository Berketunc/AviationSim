#!/usr/bin/env python3
"""Generate the final Isaac and Gazebo comparison figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTROLLERS = ("Classical", "Standalone RL", "Residual RL")
COLORS = ("#4C78A8", "#F58518", "#54A24B")


def _read_csv(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["controller"]: {
            key: float(value) for key, value in row.items() if key != "controller"
        }
        for row in rows
    }


def _annotate(ax, bars, *, percent: bool = False) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{100.0 * value:.1f}%" if percent else f"{value:.3f}"
        ax.annotate(
            label,
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90 if percent else 0,
            fontsize=8,
        )


def plot_isaac(results_dir: Path, output_dir: Path) -> Path:
    profile_files = {
        "Nominal": results_dir / "nominal_comparison.csv",
        "Moderate": results_dir / "moderate_robustness_comparison.csv",
        "Severe": results_dir / "severe_robustness_comparison.csv",
    }
    profiles = {name: _read_csv(path) for name, path in profile_files.items()}
    x = np.arange(len(profile_files))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for index, (controller, color) in enumerate(zip(CONTROLLERS, COLORS)):
        success = [profiles[name][controller]["success_rate"] for name in profile_files]
        efficiency = []
        for name in profile_files:
            row = profiles[name][controller]
            if "reliability_adjusted_efficiency" in row:
                efficiency.append(row["reliability_adjusted_efficiency"])
            else:
                efficiency.append(
                    row["success_rate"] * row["trajectory_efficiency"]
                )
        offset = (index - 1) * width
        bars = axes[0].bar(x + offset, success, width, label=controller, color=color)
        _annotate(axes[0], bars, percent=True)
        axes[1].bar(x + offset, efficiency, width, label=controller, color=color)

    for ax in axes:
        ax.set_xticks(x, profile_files.keys())
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Isaac Lab completion reliability")
    axes[0].set_ylabel("Success rate")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].set_title("Reliability-adjusted trajectory efficiency")
    axes[1].set_ylabel("Success rate × successful efficiency")
    path = output_dir / "isaac_controller_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gazebo(gazebo_dir: Path, output_dir: Path) -> Path:
    files = (
        gazebo_dir / "classical_smoke_v3.json",
        gazebo_dir / "residual_smoke_v3.json",
        gazebo_dir / "residual_smoke_v4.json",
        gazebo_dir / "residual_failclosed_v5.json",
    )
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    labels = (
        "Classical\nv3",
        "Residual\nv3",
        "Residual v4\nunsafe",
        "Residual v5\nfail-closed",
    )
    colors = (COLORS[0], COLORS[2], "#E45756", "#9467BD")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    metrics = (
        ("final_distance_to_goal_m", "Final distance to goal (m)"),
        ("min_clearance_m", "Minimum clearance (m)"),
        ("duration_s", "Navigation duration (s)"),
    )
    for ax, (key, title) in zip(axes, metrics):
        values = [float(run[key]) for run in runs]
        bars = ax.bar(labels, values, color=colors)
        _annotate(ax, bars)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=0)
    for index, run in enumerate(runs):
        axes[2].text(
            index,
            float(run["duration_s"]) * 0.52,
            run["outcome"].replace("_", " "),
            ha="center",
            va="center",
            rotation=90,
            color="white",
            fontweight="bold",
            fontsize=9,
        )
    path = output_dir / "gazebo_transfer_smoke.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_gazebo_v10_pair(gazebo_dir: Path, output_dir: Path) -> Path:
    files = (
        gazebo_dir / "classical_landing_v10.json",
        gazebo_dir / "residual_landing_v10.json",
    )
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    labels = ("Classical", "Residual-assisted")
    colors = (COLORS[0], COLORS[2])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), constrained_layout=True)
    metrics = (
        ("duration_s", "Navigation duration (s)"),
        ("path_length_m", "Path length (m)"),
        ("min_clearance_m", "Minimum clearance (m)"),
    )
    for ax, (key, title) in zip(axes, metrics):
        values = [float(run[key]) for run in runs]
        bars = ax.bar(labels, values, color=colors, width=0.6)
        _annotate(ax, bars)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Gazebo/PX4 v10: both navigation and precision landing succeeded",
        fontsize=12,
    )
    path = output_dir / "gazebo_v10_pair.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_dir / "results" / "plots",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = project_dir / "results" / "evaluations"
    gazebo_dir = project_dir / "results" / "gazebo_transfer"
    for path in (
        plot_isaac(evaluation_dir, output_dir),
        plot_gazebo(gazebo_dir, output_dir),
        plot_gazebo_v10_pair(gazebo_dir, output_dir),
    ):
        print(path)


if __name__ == "__main__":
    main()
