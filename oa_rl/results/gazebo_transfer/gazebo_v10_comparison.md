# Gazebo/PX4 v10 Paired Smoke Comparison

These are one classical and one residual-assisted end-to-end run from
equivalent start conditions. They are diagnostic paired smoke trials, not
a statistically powered benchmark.

| Controller | Navigation | Landing | Duration (s) | Path (m) | Final distance (m) | Min clearance (m) | Mean latency (ms) | Residual samples | Fallback |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Classical | success | success | 76.520 | 18.198 | 0.265 | 0.711 | 0.035 | 0/1485 | False |
| Residual | success | success | 74.180 | 17.988 | 0.257 | 0.682 | 0.294 | 679/1462 | True |

## Residual-assisted versus classical

- Duration: -2.340 (-3.1%).
- Path length: -0.210 (-1.2%).
- Final goal distance: -0.008 (-3.0%).
- Minimum clearance: -0.029 (-4.1%).
- Mean controller latency: +0.259 (+746.4%).
- Mean applied residual: 0.095 m/s.
- Clearance-shield interventions: 189.
- Residual inference was used for 46.4% of controller samples, then an empty replan triggered permanent fail-closed fallback.

Both runs completed navigation and precision landing. The residual run
validates the bounded fail-closed hybrid pipeline, not uninterrupted direct
policy transfer: the policy was inactive at mission completion.

## Sources

- Classical: `results/gazebo_transfer/classical_landing_v10.json`
- Residual-assisted: `results/gazebo_transfer/residual_landing_v10.json`
