"""Portable NumPy inference and safety bounds for the residual policy.

This module deliberately has no Isaac Lab, RSL-RL, Torch, ROS, or MAVSDK
dependency. The ROS follower supplies world-frame state and the classical
velocity proposal; this module reproduces the trained observation ordering and
returns a bounded corrected world-frame planar velocity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


OBSERVATION_DIM = 22
ACTION_DIM = 2
NEAREST_PILLAR_COUNT = 6

PILLAR_XY = np.asarray(
    [
        (-5.0, -6.0),
        (-5.0, -3.0),
        (-5.0, 0.0),
        (-5.0, 3.0),
        (-5.0, 6.0),
        (0.0, -4.5),
        (0.0, -1.5),
        (0.0, 1.5),
        (0.0, 4.5),
        (5.0, -6.0),
        (5.0, -3.0),
        (5.0, 0.0),
        (5.0, 3.0),
        (5.0, 6.0),
    ],
    dtype=np.float32,
)
WALL_BOUNDS = (-10.0, 10.0, -7.0, 7.0)

_EXPECTED_WEIGHT_SHAPES = {
    "weight_0": (64, 22),
    "bias_0": (64,),
    "weight_1": (64, 64),
    "bias_1": (64,),
    "weight_2": (2, 64),
    "bias_2": (2,),
}


class ResidualPolicyError(RuntimeError):
    """Raised when policy loading or inference cannot be trusted."""


def _vector(name: str, value: Iterable[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (2,):
        raise ResidualPolicyError(f"{name} must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ResidualPolicyError(f"{name} contains a non-finite value")
    return array


def build_observation(
    position_xy: Iterable[float],
    velocity_world_xy: Iterable[float],
    goal_xy: Iterable[float],
    classical_velocity_world_xy: Iterable[float],
    *,
    velocity_scale: float = 1.0,
) -> np.ndarray:
    """Build the exact 22-value residual observation used during training.

    velocity_scale maps deployment velocities back to the training speed
    scale. For the Gazebo follower's 0.4 m/s cruise speed and the training
    controller's 1.0 m/s speed, pass 0.4. Geometry remains in metres.
    """

    if not np.isfinite(velocity_scale) or velocity_scale <= 0.0:
        raise ResidualPolicyError("velocity_scale must be finite and positive")

    position = _vector("position_xy", position_xy)
    velocity = _vector("velocity_world_xy", velocity_world_xy)
    goal = _vector("goal_xy", goal_xy)
    classical_velocity = _vector(
        "classical_velocity_world_xy", classical_velocity_world_xy
    )

    pillar_relative = PILLAR_XY - position
    squared_distance = np.sum(pillar_relative * pillar_relative, axis=1)
    # Stable sorting gives deterministic ordering for exactly equidistant
    # pillars; ordinary states are ordered strictly by Euclidean distance.
    nearest_indices = np.argsort(squared_distance, kind="stable")[
        :NEAREST_PILLAR_COUNT
    ]
    nearest_relative = pillar_relative[nearest_indices].reshape(-1)

    west_x, east_x, south_y, north_y = WALL_BOUNDS
    wall_clearance = np.asarray(
        [
            position[0] - west_x,
            east_x - position[0],
            position[1] - south_y,
            north_y - position[1],
        ],
        dtype=np.float32,
    )

    observation = np.concatenate(
        [
            goal - position,
            velocity / np.float32(velocity_scale),
            classical_velocity / np.float32(velocity_scale),
            nearest_relative,
            wall_clearance,
        ]
    ).astype(np.float32, copy=False)
    if observation.shape != (OBSERVATION_DIM,):
        raise ResidualPolicyError(
            f"observation must have shape ({OBSERVATION_DIM},), got {observation.shape}"
        )
    if not np.all(np.isfinite(observation)):
        raise ResidualPolicyError("observation contains a non-finite value")
    return observation


class ResidualPolicy:
    """Deterministic 22->64->64->2 ELU actor loaded from a NumPy bundle."""

    def __init__(self, weights_path: str | Path):
        self.weights_path = Path(weights_path).expanduser().resolve()
        if not self.weights_path.is_file():
            raise ResidualPolicyError(f"policy weights not found: {self.weights_path}")

        try:
            with np.load(self.weights_path, allow_pickle=False) as bundle:
                names = set(bundle.files)
                if names != set(_EXPECTED_WEIGHT_SHAPES):
                    raise ResidualPolicyError(
                        "unexpected policy tensors: "
                        f"expected {sorted(_EXPECTED_WEIGHT_SHAPES)}, got {sorted(names)}"
                    )
                self._weights = {
                    name: np.asarray(bundle[name], dtype=np.float32).copy()
                    for name in _EXPECTED_WEIGHT_SHAPES
                }
        except ResidualPolicyError:
            raise
        except Exception as exc:
            raise ResidualPolicyError(
                f"failed to load policy weights {self.weights_path}: {exc}"
            ) from exc

        for name, expected_shape in _EXPECTED_WEIGHT_SHAPES.items():
            array = self._weights[name]
            if array.shape != expected_shape:
                raise ResidualPolicyError(
                    f"{name} has shape {array.shape}, expected {expected_shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ResidualPolicyError(f"{name} contains a non-finite value")

    @staticmethod
    def _elu(value: np.ndarray) -> np.ndarray:
        output = value.astype(np.float32, copy=True)
        negative = output <= 0.0
        output[negative] = np.expm1(output[negative])
        return output

    def predict(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim not in (1, 2) or obs.shape[-1] != OBSERVATION_DIM:
            raise ResidualPolicyError(
                f"observation must end in dimension {OBSERVATION_DIM}, got {obs.shape}"
            )
        if not np.all(np.isfinite(obs)):
            raise ResidualPolicyError("observation contains a non-finite value")

        hidden_0 = self._elu(
            obs @ self._weights["weight_0"].T + self._weights["bias_0"]
        )
        hidden_1 = self._elu(
            hidden_0 @ self._weights["weight_1"].T + self._weights["bias_1"]
        )
        action = (
            hidden_1 @ self._weights["weight_2"].T + self._weights["bias_2"]
        ).astype(np.float32, copy=False)
        if not np.all(np.isfinite(action)):
            raise ResidualPolicyError("policy produced a non-finite action")
        return action

    def correct_velocity(
        self,
        observation: np.ndarray,
        classical_velocity_world_xy: Iterable[float],
        *,
        residual_scale_mps: float,
        combined_speed_cap_mps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return corrected velocity, residual velocity, and clipped action."""

        if not np.isfinite(residual_scale_mps) or residual_scale_mps < 0.0:
            raise ResidualPolicyError(
                "residual_scale_mps must be finite and non-negative"
            )
        if not np.isfinite(combined_speed_cap_mps) or combined_speed_cap_mps <= 0.0:
            raise ResidualPolicyError(
                "combined_speed_cap_mps must be finite and positive"
            )

        classical_velocity = _vector(
            "classical_velocity_world_xy", classical_velocity_world_xy
        )
        action = np.clip(self.predict(observation), -1.0, 1.0)
        if action.shape != (ACTION_DIM,):
            raise ResidualPolicyError(
                f"single observation must produce shape ({ACTION_DIM},), got {action.shape}"
            )
        residual_velocity = action * np.float32(residual_scale_mps)
        corrected_velocity = classical_velocity + residual_velocity

        speed = float(np.linalg.norm(corrected_velocity))
        if speed > combined_speed_cap_mps:
            corrected_velocity *= np.float32(combined_speed_cap_mps / speed)
        return corrected_velocity, residual_velocity, action
