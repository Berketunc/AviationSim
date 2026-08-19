#!/usr/bin/env python3
"""Deterministic evaluation harness for AviationSim warehouse controllers.

Evaluates one of three controller modes through the same Isaac Lab scene and
termination logic:
  classical  -- GoalFlowField only, no learned policy
  standalone -- the pre-residual direct-action policy (20 observations)
  residual   -- GoalFlowField plus a learned residual (22 observations)

Writes a JSON summary and per-episode CSV under oa_rl/results/evaluations/.
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate AviationSim warehouse controllers.")
parser.add_argument("--task", default="Isaac-WarehouseAvoidance-Direct-v0")
parser.add_argument("--mode", choices=("classical", "standalone", "residual"), required=True)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--episodes", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--benchmark_iterations", type=int, default=500)
parser.add_argument("--output_dir", type=str, default="results/evaluations")
parser.add_argument(
    "--robustness_profile",
    choices=("nominal", "moderate", "severe"),
    default="nominal",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.mode != "classical" and not args_cli.checkpoint:
    parser.error("--checkpoint is required for standalone and residual modes")
if args_cli.mode == "classical" and args_cli.checkpoint:
    parser.error("--checkpoint is not used in classical mode")
if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
    parser.error("--episodes and --num_envs must be positive")
if args_cli.episodes != args_cli.num_envs:
    parser.error(
        "paired evaluation requires --episodes to equal --num_envs so exactly "
        "one first episode is collected from every environment"
    )

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import oa_rl.tasks  # noqa: F401

RSL_RL_VERSION = metadata.version("rsl-rl-lib")

ROBUSTNESS_PROFILES = {
    "nominal": {
        "spawn_jitter_xy_m": (0.0, 0.0),
        "actuator_gain_range": (1.0, 1.0),
        "wind_velocity_max_mps": 0.0,
        "observation_noise_std": 0.0,
    },
    "moderate": {
        "spawn_jitter_xy_m": (0.25, 0.75),
        "actuator_gain_range": (0.85, 1.15),
        "wind_velocity_max_mps": 0.10,
        "observation_noise_std": 0.02,
    },
    "severe": {
        "spawn_jitter_xy_m": (0.50, 1.25),
        "actuator_gain_range": (0.70, 1.30),
        "wind_velocity_max_mps": 0.20,
        "observation_noise_std": 0.05,
    },
}


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "p50": torch.quantile(values, 0.50).item(),
        "p95": torch.quantile(values, 0.95).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def benchmark_ms(fn, iterations: int, device: str) -> dict[str, float]:
    for _ in range(50):
        fn()
    if "cuda" in device:
        torch.cuda.synchronize()
        pairs = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iterations)
        ]
        for start, end in pairs:
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        samples = [start.elapsed_time(end) for start, end in pairs]
    else:
        samples = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - started) / 1.0e6)
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "iterations": iterations,
    }


def checkpoint_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def evaluation_source_digest() -> str:
    """Hash the evaluator and installed-project source needed for a rollout."""
    paths = [Path(__file__).resolve()]
    paths.extend(sorted((PROJECT_ROOT / "source" / "oa_rl").rglob("*.py")))
    paths.extend(
        path
        for path in (
            PROJECT_ROOT / "source" / "oa_rl" / "pyproject.toml",
            PROJECT_ROOT / "source" / "oa_rl" / "setup.py",
        )
        if path.is_file()
    )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(PROJECT_ROOT)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_provenance() -> dict[str, object]:
    """Capture both the base commit and whether that commit fully describes the run."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status_lines = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except subprocess.CalledProcessError:
        return {"commit": None, "dirty": None, "status_porcelain": None}
    return {
        "commit": commit,
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
    }


def installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def phase(message: str):
    print(f"[eval] {message}", flush=True)


def main():
    # If an Isaac/torch call stalls, emit the live Python stack instead of
    # leaving a silent process that has to be guessed at externally.
    faulthandler.dump_traceback_later(60, repeat=True)
    phase("parsing environment configuration")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.collect_evaluation_metrics = True
    env_cfg.randomize_initial_episode_length = False
    env_cfg.debug_vis = False
    robustness_cfg = ROBUSTNESS_PROFILES[args_cli.robustness_profile]
    for key, value in robustness_cfg.items():
        setattr(env_cfg, key, value)

    if args_cli.mode == "standalone":
        env_cfg.standalone_policy_action = True
        env_cfg.use_residual_action = False
        env_cfg.observation_space = 20
    elif args_cli.mode == "classical":
        env_cfg.standalone_policy_action = False
        env_cfg.use_residual_action = False
        env_cfg.observation_space = 22
    else:
        env_cfg.standalone_policy_action = False
        env_cfg.use_residual_action = True
        env_cfg.observation_space = 22

    phase("creating Isaac environment")
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    phase("Isaac environment created; loading agent configuration")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device
    phase("resetting environment through the RSL-RL wrapper")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    obs = env.get_observations()
    phase("environment reset complete")

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve() if args_cli.checkpoint else None
    policy = None
    policy_nn = None
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        load_path = handle_deprecated_rsl_rl_checkpoint(str(checkpoint), RSL_RL_VERSION)
        runner.load(load_path)
        policy = runner.get_inference_policy(device=base_env.device)
        # RSL-RL >=4 resets recurrent state through the inference policy.
        # actor_critic is only needed for the legacy (<4) reset API.
        if version.parse(RSL_RL_VERSION) < version.parse("4.0.0"):
            policy_nn = runner.alg.actor_critic

    phase("checkpoint loaded" if checkpoint is not None else "classical mode: no checkpoint")
    phase(f"benchmarking controller ({args_cli.benchmark_iterations} iterations)")
    compute = {
        "flow_field_build_ms": (
            base_env._flow_field.build_time_s * 1000.0 if base_env._flow_field is not None else 0.0
        ),
        "classical_lookup_batch1": None,
        "policy_inference_batch1": None,
    }
    if base_env._flow_field is not None:
        sample_pos = torch.tensor([[-8.5, 0.0]], device=base_env.device)
        compute["classical_lookup_batch1"] = benchmark_ms(
            lambda: base_env._flow_field.direction_at(sample_pos),
            args_cli.benchmark_iterations,
            base_env.device,
        )
    if policy is not None:
        single_obs = obs[:1]
        compute["policy_inference_batch1"] = benchmark_ms(
            lambda: policy(single_obs),
            args_cli.benchmark_iterations,
            base_env.device,
        )

    phase("controller benchmark complete; starting episode rollouts")
    records: dict[str, list[torch.Tensor]] = {}
    seen_scenario_ids: set[int] = set()
    completed = 0
    control_steps = 0
    next_progress = max(1, args_cli.episodes // 10)
    started = time.perf_counter()

    while simulation_app.is_running() and completed < args_cli.episodes:
        with torch.inference_mode():
            if policy is None:
                actions = torch.zeros((args_cli.num_envs, 2), device=base_env.device)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if policy is not None:
                if version.parse(RSL_RL_VERSION) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

        control_steps += 1
        for batch in base_env.pop_completed_episode_batches():
            first_episode = batch["episode_index"].eq(0)
            selected = [
                index
                for index in torch.nonzero(first_episode, as_tuple=False).flatten().tolist()
                if int(batch["scenario_id"][index].item()) not in seen_scenario_ids
            ]
            if not selected:
                continue
            selected_tensor = torch.tensor(selected, dtype=torch.long)
            for key, values in batch.items():
                records.setdefault(key, []).append(values[selected_tensor])
            seen_scenario_ids.update(
                int(value) for value in batch["scenario_id"][selected_tensor].tolist()
            )
            completed = len(seen_scenario_ids)
        if completed >= next_progress:
            print(f"[eval] completed {min(completed, args_cli.episodes)}/{args_cli.episodes} episodes")
            next_progress += max(1, args_cli.episodes // 10)

    wall_time_s = time.perf_counter() - started
    if completed < args_cli.episodes:
        raise RuntimeError(f"Simulation stopped after only {completed} completed episodes")

    tensors = {key: torch.cat(chunks) for key, chunks in records.items()}
    scenario_order = torch.argsort(tensors["scenario_id"])
    tensors = {key: values[scenario_order] for key, values in tensors.items()}
    expected_scenarios = torch.arange(args_cli.episodes, dtype=tensors["scenario_id"].dtype)
    if not torch.equal(tensors["scenario_id"], expected_scenarios):
        raise RuntimeError("Did not collect exactly one first episode for every scenario ID")

    summary = {
        "success_rate": tensors["goal_reached"].float().mean().item(),
        "collision_rate": tensors["collision"].float().mean().item(),
        "out_of_bounds_rate": tensors["out_of_bounds"].float().mean().item(),
        "time_out_rate": tensors["time_out"].float().mean().item(),
    }
    for key in (
        "episode_steps",
        "duration_s",
        "final_distance_to_goal_m",
        "path_length_m",
        "trajectory_efficiency",
        "command_variation_mps",
        "command_variation_rate_mps2",
        "min_clearance_m",
        "mean_residual_mps",
    ):
        summary[key] = tensor_stats(tensors[key])

    successful_mask = tensors["goal_reached"].bool()
    successful_count = int(successful_mask.sum().item())
    summary["successful_episode_count"] = successful_count
    successful_keys = (
        "duration_s",
        "final_distance_to_goal_m",
        "path_length_m",
        "trajectory_efficiency",
        "command_variation_mps",
        "command_variation_rate_mps2",
        "min_clearance_m",
        "mean_residual_mps",
    )
    summary["successful_only"] = (
        {key: tensor_stats(tensors[key][successful_mask]) for key in successful_keys}
        if successful_count
        else {}
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_suffix = (
        "" if args_cli.robustness_profile == "nominal" else f"_{args_cli.robustness_profile}"
    )
    stem = f"{timestamp}_{args_cli.mode}{profile_suffix}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}_episodes.csv"

    provenance = git_provenance()

    result = {
        "mode": args_cli.mode,
        "task": args_cli.task,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": checkpoint_digest(checkpoint),
        # git_commit is retained for compatibility with the existing result
        # schema. provenance and evaluation_source_sha256 make dirty runs
        # independently identifiable instead of attributing them only to HEAD.
        "git_commit": provenance["commit"],
        "provenance": provenance,
        "evaluation_source_sha256": evaluation_source_digest(),
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "rsl_rl_lib": RSL_RL_VERSION,
            "isaaclab": installed_version("isaaclab"),
            "isaaclab_rl": installed_version("isaaclab-rl"),
        },
        "seed": args_cli.seed,
        "robustness_profile": args_cli.robustness_profile,
        "robustness_parameters": robustness_cfg,
        "num_envs": args_cli.num_envs,
        "episodes": args_cli.episodes,
        "sampling_protocol": {
            "paired_first_episode_per_environment": True,
            "scenario_id": "environment index",
            "scenario_rng_seed": args_cli.seed + 10_003,
            "observation_noise_rng_seed": args_cli.seed + 20_003,
        },
        "step_dt_s": base_env.step_dt,
        "wall_time_s": wall_time_s,
        "control_steps": control_steps,
        "simulated_env_steps": control_steps * args_cli.num_envs,
        "simulated_env_steps_per_second": control_steps * args_cli.num_envs / wall_time_s,
        "compute": compute,
        "summary": summary,
    }
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    fieldnames = list(tensors)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        rows = zip(*(tensors[key].tolist() for key in fieldnames))
        for row in rows:
            writer.writerow(dict(zip(fieldnames, row)))

    faulthandler.cancel_dump_traceback_later()
    print("\nEvaluation summary")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Print the real failure before SimulationApp cleanup: Kit can hang
        # during close after a partially constructed scene.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
