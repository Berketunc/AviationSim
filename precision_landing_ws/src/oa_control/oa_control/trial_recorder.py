"""Deterministic navigation-trial metrics for Gazebo transfer tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from oa_control.residual_policy import PILLAR_XY


PILLAR_HALF_SIZE_M = 0.2
WALLS = (
    ((0.0, 7.0), (10.2, 0.1)),
    ((0.0, -7.0), (10.2, 0.1)),
    ((10.0, 0.0), (0.1, 7.0)),
    ((-10.0, 0.0), (0.1, 7.0)),
)


def _xy(value: Iterable[float]) -> tuple[float, float]:
    values = tuple(float(component) for component in value)
    if len(values) != 2 or not all(math.isfinite(component) for component in values):
        raise ValueError(f"expected two finite XY values, got {values}")
    return values


def _box_surface_distance(
    position_xy: tuple[float, float],
    center_xy: tuple[float, float],
    half_size_xy: tuple[float, float],
) -> float:
    dx = abs(position_xy[0] - center_xy[0]) - half_size_xy[0]
    dy = abs(position_xy[1] - center_xy[1]) - half_size_xy[1]
    outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside = min(max(dx, dy), 0.0)
    return outside + inside


def obstacle_surface_clearance(position_xy: Iterable[float]) -> float:
    """Signed distance from a point to the closest pillar or wall surface."""

    position = _xy(position_xy)
    clearances = [
        _box_surface_distance(
            position,
            (float(pillar[0]), float(pillar[1])),
            (PILLAR_HALF_SIZE_M, PILLAR_HALF_SIZE_M),
        )
        for pillar in PILLAR_XY
    ]
    clearances.extend(
        _box_surface_distance(position, center, half_size)
        for center, half_size in WALLS
    )
    return min(clearances)


def _cap_planar(
    velocity_xy: tuple[float, float], speed_cap_mps: float
) -> tuple[float, float]:
    speed = math.hypot(*velocity_xy)
    if speed <= speed_cap_mps:
        return velocity_xy
    scale = speed_cap_mps / speed
    return velocity_xy[0] * scale, velocity_xy[1] * scale


def shield_residual_velocity(
    position_xy: Iterable[float],
    classical_velocity_xy: Iterable[float],
    residual_velocity_xy: Iterable[float],
    *,
    speed_cap_mps: float,
    minimum_clearance_m: float,
    lookahead_s: float,
    clearance_release_margin_m: float = 0.35,
    maximum_blend: float = 1.0,
    blend_steps: int = 8,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Back off residual authority until projected clearance is recoverable.

    The classical proposal is always admissible. If its own projected
    clearance is below the requested margin, a residual candidate must at
    least be no worse than classical. This preserves the verified baseline
    while preventing the learned correction from entering planner-inflated
    regions that cause subsequent A* replans to publish an empty path.
    """

    position = _xy(position_xy)
    classical = _xy(classical_velocity_xy)
    residual = _xy(residual_velocity_xy)
    if not math.isfinite(speed_cap_mps) or speed_cap_mps <= 0.0:
        raise ValueError("speed_cap_mps must be finite and positive")
    if not math.isfinite(minimum_clearance_m) or minimum_clearance_m <= 0.0:
        raise ValueError("minimum_clearance_m must be finite and positive")
    if not math.isfinite(lookahead_s) or lookahead_s <= 0.0:
        raise ValueError("lookahead_s must be finite and positive")
    if (
        not math.isfinite(clearance_release_margin_m)
        or clearance_release_margin_m <= 0.0
    ):
        raise ValueError(
            "clearance_release_margin_m must be finite and positive"
        )
    if (
        not math.isfinite(maximum_blend)
        or not 0.0 <= maximum_blend <= 1.0
    ):
        raise ValueError("maximum_blend must be in [0, 1]")
    if blend_steps < 1:
        raise ValueError("blend_steps must be positive")

    classical_capped = _cap_planar(classical, speed_cap_mps)
    classical_projection = (
        position[0] + classical_capped[0] * lookahead_s,
        position[1] + classical_capped[1] * lookahead_s,
    )
    required_clearance = min(
        minimum_clearance_m,
        obstacle_surface_clearance(classical_projection),
    )

    current_clearance = obstacle_surface_clearance(position)
    clearance_blend = max(
        0.0,
        min(
            1.0,
            (current_clearance - minimum_clearance_m)
            / clearance_release_margin_m,
        ),
    )
    initial_blend = min(maximum_blend, clearance_blend)

    for index in range(blend_steps + 1):
        blend = initial_blend * (1.0 - index / blend_steps)
        candidate = _cap_planar(
            (
                classical[0] + blend * residual[0],
                classical[1] + blend * residual[1],
            ),
            speed_cap_mps,
        )
        projection = (
            position[0] + candidate[0] * lookahead_s,
            position[1] + candidate[1] * lookahead_s,
        )
        if obstacle_surface_clearance(projection) + 1.0e-6 >= required_clearance:
            applied_residual = (
                candidate[0] - classical[0],
                candidate[1] - classical[1],
            )
            return candidate, applied_residual, blend

    return classical_capped, (0.0, 0.0), 0.0



def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = 0.95 * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    fraction = rank - lower
    p95 = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return {
        "mean": sum(values) / len(values),
        "p95": p95,
        "max": ordered[-1],
    }


class NavigationTrialRecorder:
    """Accumulate one navigation trial and optionally write an atomic JSON."""

    def __init__(
        self,
        *,
        controller: str,
        label: str = "",
        output_path: str | Path | None = None,
        timeout_s: float = 90.0,
        collision_radius_m: float = 0.35,
    ):
        if controller not in ("classical", "residual"):
            raise ValueError(f"unsupported controller: {controller}")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        if not math.isfinite(collision_radius_m) or collision_radius_m <= 0.0:
            raise ValueError("collision_radius_m must be finite and positive")

        self.controller = controller
        self.label = label
        self.output_path = (
            Path(output_path).expanduser().resolve() if output_path else None
        )
        self.timeout_s = float(timeout_s)
        self.collision_radius_m = float(collision_radius_m)

        self.started_at_s: float | None = None
        self.finished_at_s: float | None = None
        self.start_position_xy: tuple[float, float] | None = None
        self._last_position_xy: tuple[float, float] | None = None
        self.goal_xy: tuple[float, float] | None = None
        self.path_length_m = 0.0
        self.min_clearance_m = math.inf
        self.collision_detected = False
        self.residual_fallback = False
        self.controller_latencies_ms: list[float] = []
        self.residual_latencies_ms: list[float] = []
        self.residual_magnitudes_mps: list[float] = []
        self.residual_shield_blends: list[float] = []
        self.residual_goal_handoff_count = 0
        self.residual_recovery_count = 0
        self.path_update_count = 0
        self.empty_path_update_count = 0
        self.result: dict | None = None

    @property
    def started(self) -> bool:
        return self.started_at_s is not None

    @property
    def finished(self) -> bool:
        return self.result is not None

    def start(self, sim_time_s: float, position_xy: Iterable[float] | None = None) -> None:
        if self.started:
            return
        if not math.isfinite(sim_time_s):
            raise ValueError("sim_time_s must be finite")
        self.started_at_s = float(sim_time_s)
        if position_xy is not None:
            self.start_position_xy = _xy(position_xy)
            self._last_position_xy = self.start_position_xy
            self._update_clearance(self._last_position_xy)

    def set_goal(self, goal_xy: Iterable[float]) -> None:
        self.goal_xy = _xy(goal_xy)

    def observe(self, sim_time_s: float, position_xy: Iterable[float]) -> None:
        if self.finished:
            return
        position = _xy(position_xy)
        if not self.started:
            self.start(sim_time_s, position)
            return
        if not math.isfinite(sim_time_s):
            raise ValueError("sim_time_s must be finite")

        if self._last_position_xy is not None:
            self.path_length_m += math.hypot(
                position[0] - self._last_position_xy[0],
                position[1] - self._last_position_xy[1],
            )
        self._last_position_xy = position
        self._update_clearance(position)

    def _update_clearance(self, position_xy: tuple[float, float]) -> None:
        clearance = obstacle_surface_clearance(position_xy)
        self.min_clearance_m = min(self.min_clearance_m, clearance)
        if clearance <= self.collision_radius_m:
            self.collision_detected = True

    def timed_out(self, sim_time_s: float) -> bool:
        return (
            self.started_at_s is not None
            and math.isfinite(sim_time_s)
            and sim_time_s - self.started_at_s >= self.timeout_s
        )

    def record_path_update(self, has_path: bool) -> None:
        self.path_update_count += 1
        if not has_path:
            self.empty_path_update_count += 1

    def record_residual_recovery(self) -> None:
        self.residual_recovery_count += 1

    def record_controller_latency(self, latency_ns: int) -> None:
        if latency_ns >= 0:
            self.controller_latencies_ms.append(latency_ns / 1.0e6)

    def record_residual_inference(
        self,
        latency_ns: int,
        magnitude_mps: float,
        shield_blend: float = 1.0,
        goal_handoff: bool = False,
    ) -> None:
        if (
            latency_ns >= 0
            and math.isfinite(magnitude_mps)
            and math.isfinite(shield_blend)
            and 0.0 <= shield_blend <= 1.0
        ):
            self.residual_latencies_ms.append(latency_ns / 1.0e6)
            self.residual_magnitudes_mps.append(float(magnitude_mps))
            self.residual_shield_blends.append(float(shield_blend))
            if goal_handoff:
                self.residual_goal_handoff_count += 1

    def mark_residual_fallback(self) -> None:
        self.residual_fallback = True

    def record_landing_outcome(self, outcome: str) -> None:
        if self.result is None:
            raise RuntimeError(
                "navigation result must exist before recording landing"
            )
        if outcome not in (
            "in_progress",
            "success",
            "marker_not_found",
            "land_failed",
            "aborted",
        ):
            raise ValueError(f"unsupported landing outcome: {outcome}")
        self.result["landing_outcome"] = outcome
        self._write_result()

    def _write_result(self) -> None:
        if self.output_path is None or self.result is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(
            self.output_path.suffix + ".tmp"
        )
        temporary_path.write_text(json.dumps(self.result, indent=2) + "\n")
        temporary_path.replace(self.output_path)

    def finish(
        self,
        outcome: str,
        sim_time_s: float,
        *,
        residual_active_at_finish: bool,
    ) -> dict:
        if self.result is not None:
            return self.result
        if outcome not in ("success", "collision", "time_out", "aborted"):
            raise ValueError(f"unsupported outcome: {outcome}")
        if not self.started:
            self.start(sim_time_s)
        self.finished_at_s = float(sim_time_s)

        duration_s = max(0.0, self.finished_at_s - self.started_at_s)
        min_clearance = (
            self.min_clearance_m if math.isfinite(self.min_clearance_m) else None
        )
        final_distance_to_goal_m = (
            math.hypot(
                self.goal_xy[0] - self._last_position_xy[0],
                self.goal_xy[1] - self._last_position_xy[1],
            )
            if self.goal_xy is not None and self._last_position_xy is not None
            else None
        )
        self.result = {
            "schema_version": 5,
            "label": self.label,
            "controller": self.controller,
            "outcome": outcome,
            "success": outcome == "success",
            "collision": outcome == "collision" or self.collision_detected,
            "time_out": outcome == "time_out",
            "landing_outcome": (
                "in_progress" if outcome == "success" else "not_started"
            ),
            "duration_s": duration_s,
            "path_length_m": self.path_length_m,
            "start_position_xy": (
                list(self.start_position_xy)
                if self.start_position_xy is not None
                else None
            ),
            "final_position_xy": (
                list(self._last_position_xy)
                if self._last_position_xy is not None
                else None
            ),
            "goal_xy": list(self.goal_xy) if self.goal_xy is not None else None,
            "final_distance_to_goal_m": final_distance_to_goal_m,
            "min_clearance_m": min_clearance,
            "collision_radius_m": self.collision_radius_m,
            "timeout_s": self.timeout_s,
            "residual_active_at_finish": bool(residual_active_at_finish),
            "residual_fallback": self.residual_fallback,
            "controller_latency_ms": _summary(self.controller_latencies_ms),
            "residual_inference_latency_ms": _summary(
                self.residual_latencies_ms
            ),
            "mean_residual_mps": (
                sum(self.residual_magnitudes_mps)
                / len(self.residual_magnitudes_mps)
                if self.residual_magnitudes_mps
                else 0.0
            ),
            "mean_residual_shield_blend": (
                sum(self.residual_shield_blends)
                / len(self.residual_shield_blends)
                if self.residual_shield_blends
                else None
            ),
            "shield_intervention_count": sum(
                blend < 1.0 - 1.0e-6
                for blend in self.residual_shield_blends
            ),
            "residual_goal_handoff_count": self.residual_goal_handoff_count,
            "residual_recovery_count": self.residual_recovery_count,
            "path_updates": {
                "total": self.path_update_count,
                "nonempty": (
                    self.path_update_count - self.empty_path_update_count
                ),
                "empty": self.empty_path_update_count,
            },
            "sample_counts": {
                "controller": len(self.controller_latencies_ms),
                "residual": len(self.residual_latencies_ms),
            },
        }
        self._write_result()
        return self.result
