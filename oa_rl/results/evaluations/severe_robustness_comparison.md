# Severe Robustness Comparison

Trajectory metrics are conditioned on successful episodes. Reliability-adjusted
efficiency is success rate multiplied by successful-trajectory efficiency, so an
early collision cannot appear artificially efficient.
All controllers used the same 1024 scenario IDs and exactly one first episode per environment.

| Controller | Success | Collision | Successful time (s) | Successful path (m) | Successful efficiency | Reliability-adjusted efficiency | Command variation (m/s²) | Min clearance (m) | Mean residual (m/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical | 60.1% | 0.0% | 20.870 | 20.968 | 0.8084 | 0.4855 | 9.380 | 0.375 | 0.000 |
| Standalone RL | 81.4% | 18.4% | 14.363 | 20.453 | 0.8183 | 0.6665 | 9.782 | 0.325 | 0.000 |
| Residual RL | 99.9% | 0.0% | 13.167 | 17.772 | 0.9423 | 0.9413 | 3.004 | 0.414 | 0.552 |

## Source evaluations

- Classical: `results/evaluations/severe_full_seed42/2026-08-19_10-33-59_classical_severe.json`
- Standalone RL: `results/evaluations/severe_full_seed42/2026-08-19_10-35-22_standalone_severe.json`
- Residual RL: `results/evaluations/severe_full_seed42/2026-08-19_10-36-27_residual_severe.json`
