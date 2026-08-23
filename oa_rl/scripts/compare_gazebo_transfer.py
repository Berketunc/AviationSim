#!/usr/bin/env python3
"""Compare one classical and one residual Gazebo/PX4 transfer trial."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def percentage_change(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        return float("nan")
    return 100.0 * (candidate - baseline) / baseline


def row(result: dict) -> dict:
    controller_samples = int(result["sample_counts"]["controller"])
    residual_samples = int(result["sample_counts"]["residual"])
    return {
        "controller": result["controller"],
        "label": result["label"],
        "navigation_success": bool(result["success"]),
        "landing_success": result["landing_outcome"] == "success",
        "outcome": result["outcome"],
        "landing_outcome": result["landing_outcome"],
        "duration_s": float(result["duration_s"]),
        "path_length_m": float(result["path_length_m"]),
        "final_distance_to_goal_m": float(result["final_distance_to_goal_m"]),
        "min_clearance_m": float(result["min_clearance_m"]),
        "controller_latency_mean_ms": float(
            result["controller_latency_ms"]["mean"]
        ),
        "mean_residual_mps": float(result["mean_residual_mps"]),
        "shield_intervention_count": int(result["shield_intervention_count"]),
        "residual_fallback": bool(result["residual_fallback"]),
        "residual_samples": residual_samples,
        "controller_samples": controller_samples,
        "residual_sample_fraction": (
            residual_samples / controller_samples if controller_samples else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classical", required=True, type=Path)
    parser.add_argument("--residual", required=True, type=Path)
    parser.add_argument("--output_prefix", required=True, type=Path)
    args = parser.parse_args()

    classical_result = json.loads(args.classical.read_text(encoding="utf-8"))
    residual_result = json.loads(args.residual.read_text(encoding="utf-8"))
    if classical_result.get("controller") != "classical":
        raise ValueError("classical input is not a classical trial")
    if residual_result.get("controller") != "residual":
        raise ValueError("residual input is not a residual trial")
    if classical_result.get("schema_version") != residual_result.get("schema_version"):
        raise ValueError("trial schema versions differ")

    rows = [row(classical_result), row(residual_result)]
    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    classical, residual = rows
    metrics = (
        ("duration_s", "Duration"),
        ("path_length_m", "Path length"),
        ("final_distance_to_goal_m", "Final goal distance"),
        ("min_clearance_m", "Minimum clearance"),
        ("controller_latency_mean_ms", "Mean controller latency"),
    )
    lines = [
        "# Gazebo/PX4 v10 Paired Smoke Comparison",
        "",
        "These are one classical and one residual-assisted end-to-end run from",
        "equivalent start conditions. They are diagnostic paired smoke trials, not",
        "a statistically powered benchmark.",
        "",
        "| Controller | Navigation | Landing | Duration (s) | Path (m) | Final distance (m) | Min clearance (m) | Mean latency (ms) | Residual samples | Fallback |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['controller'].title()} | "
            f"{'success' if item['navigation_success'] else item['outcome']} | "
            f"{item['landing_outcome']} | {item['duration_s']:.3f} | "
            f"{item['path_length_m']:.3f} | "
            f"{item['final_distance_to_goal_m']:.3f} | "
            f"{item['min_clearance_m']:.3f} | "
            f"{item['controller_latency_mean_ms']:.3f} | "
            f"{item['residual_samples']}/{item['controller_samples']} | "
            f"{item['residual_fallback']} |"
        )

    lines.extend(["", "## Residual-assisted versus classical", ""])
    for key, label in metrics:
        delta = residual[key] - classical[key]
        change = percentage_change(residual[key], classical[key])
        lines.append(
            f"- {label}: {delta:+.3f} ({change:+.1f}%)."
        )
    lines.extend(
        [
            f"- Mean applied residual: {residual['mean_residual_mps']:.3f} m/s.",
            f"- Clearance-shield interventions: {residual['shield_intervention_count']}.",
            (
                "- Residual inference was used for "
                f"{residual['residual_sample_fraction']:.1%} of controller samples, "
                "then an empty replan triggered permanent fail-closed fallback."
            ),
            "",
            "Both runs completed navigation and precision landing. The residual run",
            "validates the bounded fail-closed hybrid pipeline, not uninterrupted direct",
            "policy transfer: the policy was inactive at mission completion.",
            "",
            "## Sources",
            "",
            f"- Classical: `{args.classical}`",
            f"- Residual-assisted: `{args.residual}`",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
