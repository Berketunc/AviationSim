# Nominal Controller Comparison

All controllers were evaluated through the same Isaac Lab scene, termination
logic, fixed seed, and per-episode metric collector. Each row contains 1,024
completed nominal episodes. Because the nominal warehouse is deterministic,
these are repeated simulator instances rather than 1,024 distinct scenarios;
generalization is evaluated separately through domain randomization.

| Controller | Success | Collision | Time (s) | Path (m) | Efficiency | Command variation (m/s²) | Min clearance (m) | Mean residual (m/s) | Decision latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical | 100.0% | 0.0% | 20.050 | 20.051 | 0.8329 | 8.648 | 0.335 | 0.000 | 0.146 |
| Standalone RL | 100.0% | 0.0% | 13.680 | 20.521 | 0.8138 | 9.423 | 0.075 | 0.000 | 0.200 |
| Residual RL | 100.0% | 0.0% | 12.000 | 17.281 | 0.9664 | 1.167 | 0.375 | 0.522 | 0.328 |

## Residual RL versus classical

- Completion time: -40.1%
- Path length: -13.8%
- Trajectory efficiency: +16.0%
- Command variation rate: -86.5%
- Minimum clearance: +11.9%
- Decision latency: +124.1% (+0.181 ms absolute)

The residual controller is the nominal winner: it preserves 100% success
while completing faster, following a shorter and smoother path, and retaining
slightly more obstacle clearance. Its relative compute overhead is large only
because the classical lookup is extremely cheap; total measured decision time
is 0.328 ms, below 1% of the 40 ms control period.

## Source evaluations

- Classical: `results/evaluations/2026-08-18_09-06-27_classical.json`
- Standalone RL: `results/evaluations/2026-08-18_08-43-10_standalone.json`
- Residual RL: `results/evaluations/2026-08-18_08-44-40_residual.json`
