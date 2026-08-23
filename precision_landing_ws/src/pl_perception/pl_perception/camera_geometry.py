"""Coordinate transforms for the downward Gazebo camera."""

from __future__ import annotations

import math
from collections.abc import Sequence


def optical_translation_to_body_frd(
    translation: Sequence[float],
) -> tuple[float, float, float]:
    """Convert OpenCV optical XYZ into PX4 body forward-right-down.

    Gazebo's camera looks along its local +X axis. With the camera pitched
    +pi/2 in the vehicle SDF, optical +Z points body-down, optical +X points
    body-right, and optical +Y points body-backward.
    """

    values = tuple(float(value) for value in translation)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("translation must contain three finite values")
    x_optical, y_optical, z_optical = values
    return -y_optical, x_optical, z_optical
