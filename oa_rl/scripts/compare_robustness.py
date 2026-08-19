#!/usr/bin/env python3
"""Build a success-aware robustness comparison from evaluator artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def episode_csv_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}_episodes.csv")


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "1.0", "true"}


def episode_rows(result_path: Path) -> list[dict[str, str]]:
    csv_path = episode_csv_path(result_path)
    with csv_path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def successful_mean(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return float("nan")
    return statistics.fmean(float(row[key]) for row in rows)


def controller_latency_ms(result: dict) -> float:
    compute = result["compute"]
    latency = 0.0
    if compute["classical_lookup_batch1"] is not None:
        latency += float(compute["classical_lookup_batch1"]["mean_ms"])
    if compute["policy_inference_batch1"] is not None:
        latency += float(compute["policy_inference_batch1"]["mean_ms"])
    return latency


def validate_comparability(
    sources: dict[str, Path], results: dict[str, dict]
) -> tuple[dict[str, list[dict[str, str]]], bool]:
    expected_modes = {
        "Classical": "classical",
        "Standalone RL": "standalone",
        "Residual RL": "residual",
    }
    for label, expected_mode in expected_modes.items():
        if results[label].get("mode") != expected_mode:
            raise ValueError(
                f"{label} input has mode={results[label].get('mode')!r}, "
                f"expected {expected_mode!r}"
            )

    comparable_fields = (
        "task",
        "seed",
        "robustness_profile",
        "robustness_parameters",
        "num_envs",
        "episodes",
        "evaluation_source_sha256",
        "software_versions",
        "sampling_protocol",
    )
    reference_label = "Classical"
    reference = results[reference_label]
    for label, result in results.items():
        for field in comparable_fields:
            if result.get(field) != reference.get(field):
                raise ValueError(
                    f"{label} differs from {reference_label} in {field}: "
                    f"{result.get(field)!r} != {reference.get(field)!r}"
                )

    rows_by_controller = {
        label: episode_rows(path) for label, path in sources.items()
    }
    for label, rows in rows_by_controller.items():
        if len(rows) != int(results[label]["episodes"]):
            raise ValueError(
                f"{label} CSV has {len(rows)} episodes; "
                f"JSON declares {results[label]['episodes']}"
            )

    id_presence = {
        label: bool(rows) and "scenario_id" in rows[0]
        for label, rows in rows_by_controller.items()
    }
    if len(set(id_presence.values())) != 1:
        raise ValueError("Only some inputs contain paired scenario IDs")
    if not next(iter(id_presence.values())):
        return rows_by_controller, False

    scenario_keys = ("spawn_x_m", "spawn_y_m", "actuator_gain", "wind_x_mps", "wind_y_mps")
    reference_scenarios = None
    for label, rows in rows_by_controller.items():
        scenario_ids = [int(float(row["scenario_id"])) for row in rows]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError(f"{label} contains duplicate scenario IDs")
        if any(int(float(row["episode_index"])) != 0 for row in rows):
            raise ValueError(f"{label} contains a non-first episode")
        scenarios = {
            int(float(row["scenario_id"])): tuple(float(row[key]) for key in scenario_keys)
            for row in rows
        }
        if reference_scenarios is None:
            reference_scenarios = scenarios
        elif scenarios != reference_scenarios:
            raise ValueError(f"{label} does not use the same paired perturbations")
    return rows_by_controller, True


def make_row(
    label: str, result: dict, all_episodes: list[dict[str, str]]
) -> dict[str, float | str]:
    completed = [row for row in all_episodes if as_bool(row["goal_reached"])]
    success_rate = float(result["summary"]["success_rate"])
    efficiency = successful_mean(completed, "trajectory_efficiency")
    return {
        "controller": label,
        "success_rate": success_rate,
        "collision_rate": float(result["summary"]["collision_rate"]),
        "out_of_bounds_rate": float(result["summary"]["out_of_bounds_rate"]),
        "time_out_rate": float(result["summary"]["time_out_rate"]),
        "successful_episodes": len(completed),
        "successful_duration_s": successful_mean(completed, "duration_s"),
        "successful_path_length_m": successful_mean(completed, "path_length_m"),
        "successful_trajectory_efficiency": efficiency,
        "reliability_adjusted_efficiency": success_rate * efficiency,
        "successful_command_variation_rate_mps2": successful_mean(
            completed, "command_variation_rate_mps2"
        ),
        "successful_min_clearance_m": successful_mean(completed, "min_clearance_m"),
        "successful_mean_residual_mps": successful_mean(completed, "mean_residual_mps"),
        "decision_latency_ms": controller_latency_ms(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", required=True)
    parser.add_argument("--standalone", required=True)
    parser.add_argument("--residual", required=True)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()

    sources = {
        "Classical": Path(args.classical),
        "Standalone RL": Path(args.standalone),
        "Residual RL": Path(args.residual),
    }
    results = {name: json.loads(path.read_text()) for name, path in sources.items()}
    episodes_by_controller, paired = validate_comparability(sources, results)
    profiles = {result.get("robustness_profile", "nominal") for result in results.values()}
    if len(profiles) != 1:
        raise ValueError(f"Input evaluations use different robustness profiles: {sorted(profiles)}")
    profile = profiles.pop()
    rows = [make_row(name, results[name], episodes_by_controller[name]) for name in sources]

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {profile.title()} Robustness Comparison",
        "",
        "Trajectory metrics are conditioned on successful episodes. Reliability-adjusted",
        "efficiency is success rate multiplied by successful-trajectory efficiency, so an",
        "early collision cannot appear artificially efficient.",
        (
            f"All controllers used the same {len(episodes_by_controller['Classical'])} "
            "scenario IDs and exactly one first episode per environment."
            if paired
            else "Legacy inputs do not contain scenario IDs; pairing could not be verified."
        ),
        "",
        "| Controller | Success | Collision | Successful time (s) | Successful path (m) | Successful efficiency | Reliability-adjusted efficiency | Command variation (m/s²) | Min clearance (m) | Mean residual (m/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['controller']} | {row['success_rate']:.1%} | {row['collision_rate']:.1%} | "
            f"{row['successful_duration_s']:.3f} | {row['successful_path_length_m']:.3f} | "
            f"{row['successful_trajectory_efficiency']:.4f} | "
            f"{row['reliability_adjusted_efficiency']:.4f} | "
            f"{row['successful_command_variation_rate_mps2']:.3f} | "
            f"{row['successful_min_clearance_m']:.3f} | "
            f"{row['successful_mean_residual_mps']:.3f} |"
        )

    lines.extend(["", "## Source evaluations", ""])
    for name, path in sources.items():
        lines.append(f"- {name}: `{path}`")
    lines.append("")
    md_path.write_text("\n".join(lines))

    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
