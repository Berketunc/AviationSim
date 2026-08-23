"""Safety predicates for the final precision-landing handoff."""

from __future__ import annotations

import math


def should_handoff_to_land(
    marker_distance_m: float | None,
    odometry_altitude_m: float | None,
    threshold_m: float,
) -> bool:
    """Return true when either independent height estimate is below threshold."""

    if not math.isfinite(threshold_m) or threshold_m <= 0.0:
        raise ValueError("threshold_m must be finite and positive")
    measurements = (marker_distance_m, odometry_altitude_m)
    return any(
        value is not None
        and math.isfinite(value)
        and value >= 0.0
        and value <= threshold_m
        for value in measurements
    )
