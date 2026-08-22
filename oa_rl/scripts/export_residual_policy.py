#!/usr/bin/env python3
"""Export the selected residual actor without starting Isaac Sim.

The RSL-RL checkpoint stores the actor as a small deterministic MLP. This
script exports both TorchScript and a NumPy weight bundle, verifies that both
match the checkpoint actor, and writes the observation/action contract needed
by the Gazebo/PX4 integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


DEFAULT_CHECKPOINT = (
    "logs/rsl_rl/warehouse_avoidance_direct/"
    "2026-07-23_13-05-22_residual-penalized/model_4999.pt"
)
DEFAULT_OUTPUT_DIR = "exported_policies/residual_penalized_model_4999"

OBSERVATIONS = [
    "goal_rel_x_m",
    "goal_rel_y_m",
    "velocity_world_x_mps",
    "velocity_world_y_mps",
    "classical_velocity_world_x_mps",
    "classical_velocity_world_y_mps",
    "nearest_pillar_0_rel_x_m",
    "nearest_pillar_0_rel_y_m",
    "nearest_pillar_1_rel_x_m",
    "nearest_pillar_1_rel_y_m",
    "nearest_pillar_2_rel_x_m",
    "nearest_pillar_2_rel_y_m",
    "nearest_pillar_3_rel_x_m",
    "nearest_pillar_3_rel_y_m",
    "nearest_pillar_4_rel_x_m",
    "nearest_pillar_4_rel_y_m",
    "nearest_pillar_5_rel_x_m",
    "nearest_pillar_5_rel_y_m",
    "west_wall_clearance_m",
    "east_wall_clearance_m",
    "south_wall_clearance_m",
    "north_wall_clearance_m",
]

EXPECTED_ACTOR_SHAPES = {
    "distribution.std_param": (2,),
    "mlp.0.weight": (64, 22),
    "mlp.0.bias": (64,),
    "mlp.2.weight": (64, 64),
    "mlp.2.bias": (64,),
    "mlp.4.weight": (2, 64),
    "mlp.4.bias": (2,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verification_samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_actor(actor_state: dict[str, torch.Tensor]) -> nn.Sequential:
    actual_shapes = {key: tuple(value.shape) for key, value in actor_state.items()}
    if actual_shapes != EXPECTED_ACTOR_SHAPES:
        raise ValueError(
            "Checkpoint actor does not match the selected 22->64->64->2 ELU "
            f"architecture. Expected {EXPECTED_ACTOR_SHAPES}, got {actual_shapes}."
        )

    actor = nn.Sequential(
        nn.Linear(22, 64),
        nn.ELU(),
        nn.Linear(64, 64),
        nn.ELU(),
        nn.Linear(64, 2),
    ).eval()
    with torch.no_grad():
        for layer_index in (0, 2, 4):
            layer = actor[layer_index]
            layer.weight.copy_(actor_state[f"mlp.{layer_index}.weight"].cpu())
            layer.bias.copy_(actor_state[f"mlp.{layer_index}.bias"].cpu())
    return actor


def numpy_inference(obs: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    def elu(value: np.ndarray) -> np.ndarray:
        return np.where(value > 0.0, value, np.expm1(value)).astype(np.float32)

    hidden_0 = elu(obs @ arrays["weight_0"].T + arrays["bias_0"])
    hidden_1 = elu(hidden_0 @ arrays["weight_1"].T + arrays["bias_1"])
    return hidden_1 @ arrays["weight_2"].T + arrays["bias_2"]


def main() -> None:
    args = parse_args()
    if args.verification_samples < 1:
        raise ValueError("--verification_samples must be positive")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if "actor_state_dict" not in saved:
        raise KeyError("Checkpoint has no actor_state_dict")
    actor_state = saved["actor_state_dict"]
    actor = build_actor(actor_state)

    torchscript_path = output_dir / "residual_actor.ts"
    scripted = torch.jit.script(actor)
    scripted.save(str(torchscript_path))

    arrays = {
        "weight_0": actor_state["mlp.0.weight"].cpu().numpy(),
        "bias_0": actor_state["mlp.0.bias"].cpu().numpy(),
        "weight_1": actor_state["mlp.2.weight"].cpu().numpy(),
        "bias_1": actor_state["mlp.2.bias"].cpu().numpy(),
        "weight_2": actor_state["mlp.4.weight"].cpu().numpy(),
        "bias_2": actor_state["mlp.4.bias"].cpu().numpy(),
    }
    numpy_path = output_dir / "residual_actor_weights.npz"
    np.savez(numpy_path, **arrays)

    rng = np.random.default_rng(args.seed)
    random_obs = rng.normal(0.0, 5.0, size=(args.verification_samples, 22)).astype(np.float32)
    verification_obs = np.concatenate(
        [np.zeros((1, 22), dtype=np.float32), random_obs],
        axis=0,
    )
    obs_tensor = torch.from_numpy(verification_obs)
    with torch.inference_mode():
        checkpoint_output = actor(obs_tensor).numpy()
        scripted_output = torch.jit.load(str(torchscript_path))(obs_tensor).numpy()
    numpy_output = numpy_inference(verification_obs, arrays)

    torchscript_max_abs_error = float(np.max(np.abs(checkpoint_output - scripted_output)))
    numpy_max_abs_error = float(np.max(np.abs(checkpoint_output - numpy_output)))
    if torchscript_max_abs_error > 1.0e-6:
        raise RuntimeError(f"TorchScript verification failed: {torchscript_max_abs_error}")
    if numpy_max_abs_error > 1.0e-5:
        raise RuntimeError(f"NumPy verification failed: {numpy_max_abs_error}")

    manifest = {
        "format_version": 1,
        "policy": "penalized_residual_rl",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "iteration": int(saved.get("iter", -1)),
        },
        "architecture": {
            "input_dim": 22,
            "hidden_dims": [64, 64],
            "output_dim": 2,
            "activation": "elu",
            "observation_normalization": False,
            "deterministic_output": True,
            "dtype": "float32",
        },
        "observation_contract": {
            "frame": "warehouse world XY (Gazebo ENU axes)",
            "ordering": OBSERVATIONS,
            "nearest_pillars": {
                "count": 6,
                "ordering": "ascending Euclidean center distance",
                "coordinates": "pillar center minus vehicle position",
            },
            "wall_bounds_m": {
                "west_x": -10.0,
                "east_x": 10.0,
                "south_y": -7.0,
                "north_y": 7.0,
            },
        },
        "action_contract": {
            "network_output": "dimensionless residual action in world XY",
            "per_axis_clip": [-1.0, 1.0],
            "residual_scale_mps": 0.75,
            "combine": "classical_velocity_world_xy + clipped_action * residual_scale_mps",
            "combined_planar_speed_cap_mps": 1.5,
            "disable_behavior": "use the unmodified classical velocity command",
        },
        "validation": {
            "seed": args.seed,
            "samples": int(verification_obs.shape[0]),
            "torchscript_max_abs_error": torchscript_max_abs_error,
            "numpy_max_abs_error": numpy_max_abs_error,
        },
        "files": {
            torchscript_path.name: {
                "sha256": sha256(torchscript_path),
                "bytes": torchscript_path.stat().st_size,
            },
            numpy_path.name: {
                "sha256": sha256(numpy_path),
                "bytes": numpy_path.stat().st_size,
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Checkpoint: {checkpoint}")
    print(f"TorchScript: {torchscript_path}")
    print(f"NumPy weights: {numpy_path}")
    print(f"Manifest: {manifest_path}")
    print(
        "Verification max abs error: "
        f"TorchScript={torchscript_max_abs_error:.3e}, NumPy={numpy_max_abs_error:.3e}"
    )


if __name__ == "__main__":
    main()
