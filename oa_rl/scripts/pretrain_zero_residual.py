#!/usr/bin/env python3
"""Imitation-pretrain a residual policy to preserve the classical controller.

The teacher is the classical GoalFlowField controller. Its desired residual is
identically zero, so the actor learns the safe identity correction over states
visited by classical rollouts. The resulting checkpoint has the same native
RSL-RL structure as PPO checkpoints and can be resumed by scripts/rsl_rl/train.py.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pretrain a near-zero residual policy from classical rollouts.")
parser.add_argument("--task", default="Isaac-WarehouseAvoidance-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--updates", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=1.0e-3)
parser.add_argument("--target_action_std", type=float, default=0.10)
parser.add_argument("--validation_steps", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="zero-residual-il")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0 or args_cli.updates <= 0 or args_cli.validation_steps <= 0:
    parser.error("--num_envs, --updates, and --validation_steps must be positive")
if args_cli.learning_rate <= 0.0:
    parser.error("--learning_rate must be positive")
if args_cli.target_action_std <= 0.0:
    parser.error("--target_action_std must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import oa_rl.tasks  # noqa: F401

RSL_RL_VERSION = metadata.version("rsl-rl-lib")


def phase(message: str) -> None:
    print(f"[il] {message}", flush=True)


def set_policy_std(actor: torch.nn.Module, target_std: float) -> None:
    distribution = actor.distribution
    if hasattr(distribution, "std_param"):
        distribution.std_param.data.fill_(target_std)
    elif hasattr(distribution, "log_std_param"):
        distribution.log_std_param.data.fill_(math.log(target_std))
    else:
        raise TypeError(
            f"Unsupported policy distribution for fixed exploration std: {type(distribution).__name__}"
        )


def current_policy_std(actor: torch.nn.Module) -> list[float]:
    distribution = actor.distribution
    if hasattr(distribution, "std_param"):
        std = distribution.std_param.detach()
    elif hasattr(distribution, "log_std_param"):
        std = distribution.log_std_param.detach().exp()
    else:
        raise TypeError(f"Unsupported policy distribution: {type(distribution).__name__}")
    return std.cpu().tolist()


def project_identity_residual(actor: torch.nn.Module) -> None:
    """Set the residual output head to the exact zero-teacher optimum.

    Sequential rollout training can forget earlier parts of the trajectory.
    For an identity residual teacher the globally correct label is known to be
    zero for every possible observation, so projecting the last affine layer
    removes that closed-loop distribution-shift failure exactly.
    """
    output_layers = [module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)]
    if not output_layers:
        raise TypeError("The actor MLP has no Linear output layer to project.")
    output_layer = output_layers[-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        if output_layer.bias is not None:
            output_layer.bias.zero_()


def main() -> None:
    faulthandler.dump_traceback_later(60, repeat=True)
    torch.manual_seed(args_cli.seed)

    phase("parsing classical-rollout environment")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.use_residual_action = False
    env_cfg.standalone_policy_action = False
    env_cfg.observation_space = 22
    env_cfg.collect_evaluation_metrics = False
    env_cfg.randomize_initial_episode_length = False
    env_cfg.debug_vis = False

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        Path("logs")
        / "rsl_rl"
        / agent_cfg.experiment_name
        / f"{timestamp}_{args_cli.run_name}"
    ).resolve()
    (run_dir / "params").mkdir(parents=True, exist_ok=False)
    env_cfg.log_dir = str(run_dir)
    dump_yaml(str(run_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(run_dir / "params" / "agent.yaml"), agent_cfg)

    phase(f"creating {args_cli.num_envs} Isaac environments")
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    actor = runner.alg.actor
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args_cli.learning_rate)
    observations = env.get_observations().to(agent_cfg.device)
    zero_actions = torch.zeros((args_cli.num_envs, 2), device=env.device)

    report_every = max(1, args_cli.updates // 10)
    initial_loss = None
    final_loss = None
    phase(f"training zero-residual teacher for {args_cli.updates} updates")
    for update in range(1, args_cli.updates + 1):
        predicted_residual = actor(observations)
        loss = predicted_residual.square().mean()
        if initial_loss is None:
            initial_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = loss.item()

        with torch.inference_mode():
            observations, _, _, _ = env.step(zero_actions)
            observations = observations.to(agent_cfg.device)

        if update == 1 or update % report_every == 0 or update == args_cli.updates:
            phase(f"update {update:4d}/{args_cli.updates}: imitation_mse={final_loss:.8f}")

    # The zero teacher has a known global optimum. Projecting the final head
    # prevents a small open-loop regression error from compounding into a
    # large closed-loop trajectory shift. PPO still begins with a small but
    # nonzero exploration distribution.
    project_identity_residual(actor)
    set_policy_std(actor, args_cli.target_action_std)
    actor.eval()

    abs_sum = 0.0
    squared_sum = 0.0
    max_abs = 0.0
    action_count = 0
    with torch.inference_mode():
        for _ in range(args_cli.validation_steps):
            predicted_residual = actor(observations)
            abs_sum += predicted_residual.abs().sum().item()
            squared_sum += predicted_residual.square().sum().item()
            max_abs = max(max_abs, predicted_residual.abs().max().item())
            action_count += predicted_residual.numel()
            observations, _, _, _ = env.step(zero_actions)
            observations = observations.to(agent_cfg.device)

    validation = {
        "mean_abs_action": abs_sum / action_count,
        "rms_action": math.sqrt(squared_sum / action_count),
        "max_abs_action": max_abs,
        "mean_abs_residual_mps": abs_sum / action_count * env_cfg.residual_action_scale,
        "max_abs_residual_mps": max_abs * env_cfg.residual_action_scale,
    }
    metadata_out = {
        "method": "classical_identity_residual_imitation",
        "teacher_action": [0.0, 0.0],
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "updates": args_cli.updates,
        "samples_seen": args_cli.num_envs * args_cli.updates,
        "learning_rate": args_cli.learning_rate,
        "initial_imitation_mse": initial_loss,
        "final_imitation_mse": final_loss,
        "target_action_std": args_cli.target_action_std,
        "saved_action_std": current_policy_std(actor),
        "output_head_projected_to_exact_teacher": True,
        "validation": validation,
    }

    checkpoint = runner.alg.save()
    checkpoint["iter"] = 0
    checkpoint["infos"] = metadata_out
    checkpoint_path = run_dir / "model_il.pt"
    torch.save(checkpoint, checkpoint_path)
    (run_dir / "il_summary.json").write_text(json.dumps(metadata_out, indent=2) + "\n")

    faulthandler.cancel_dump_traceback_later()
    print("\nImitation pretraining summary")
    print(json.dumps(metadata_out, indent=2))
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Run directory: {run_dir}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
