"""Safe waypoint selection across asynchronous A* replans."""

from __future__ import annotations

import math
from collections.abc import Sequence


def select_replanned_waypoint_index(
    waypoints: Sequence[Sequence[float]],
    previous_target: Sequence[float] | None,
    *,
    maximum_preserved_index: int = 2,
    match_tolerance_m: float = 1.0e-6,
) -> int:
    """Select a nearby target without resetting progress on every replan."""

    if not waypoints:
        return 0
    default_index = 1 if len(waypoints) > 1 else 0
    if previous_target is None:
        return default_index

    last_index = min(len(waypoints) - 1, maximum_preserved_index)
    for index in range(1, last_index + 1):
        if math.dist(waypoints[index], previous_target) <= match_tolerance_m:
            return index
    return default_index
