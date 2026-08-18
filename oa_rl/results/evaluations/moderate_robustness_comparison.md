# Moderate Robustness Comparison

Trajectory metrics are conditioned on successful episodes. Reliability-adjusted
efficiency is success rate multiplied by successful-trajectory efficiency, so an
early collision cannot appear artificially efficient.

| Controller | Success | Collision | Successful time (s) | Successful path (m) | Successful efficiency | Reliability-adjusted efficiency | Command variation (m/s²) | Min clearance (m) | Mean residual (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical | 100.0% | 0.0% | 20.531 | 20.646 | 0.8136 | 0.8136 | 9.241 | 0.367 | 0.000 |
| Standalone RL | 89.7% | 10.3% | 13.629 | 20.461 | 0.8169 | 0.7331 | 9.546 | 0.233 | 0.000 |
| Residual RL | 100.0% | 0.0% | 12.616 | 17.713 | 0.9445 | 0.9445 | 2.633 | 0.410 | 0.549 |

## Source evaluations

- Classical: `results/evaluations/2026-08-18_11-28-22_classical_moderate.json`
- Standalone RL: `results/evaluations/2026-08-18_11-29-21_standalone_moderate.json`
- Residual RL: `results/evaluations/2026-08-18_11-30-32_residual_moderate.json`
