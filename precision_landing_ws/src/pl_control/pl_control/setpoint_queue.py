"""Latest-value semantics for real-time offboard control setpoints."""

from __future__ import annotations

import asyncio
from typing import TypeVar


Setpoint = TypeVar("Setpoint")


def consume_latest(queue: asyncio.Queue, current: Setpoint) -> Setpoint:
    """Drain *queue* and return its newest value, or *current* if empty.

    Velocity setpoints expire conceptually as soon as a newer command exists.
    Consuming only one FIFO entry per control tick can accumulate arbitrarily
    stale commands when producer and consumer have the same nominal rate.
    """

    while True:
        try:
            current = queue.get_nowait()
        except asyncio.QueueEmpty:
            return current
