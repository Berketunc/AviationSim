# Gazebo/PX4 transfer evidence

The directory contains chronological diagnostic trials and the validated v10
end-to-end smoke pair. These are individual simulator runs, not a statistically
powered benchmark.

## Validated v10 pair

- `classical_landing_v10.json`: classical navigation and ArUco precision
  landing both succeeded.
- `residual_landing_v10.json`: residual-assisted navigation and ArUco
  precision landing both succeeded.
- `gazebo_v10_comparison.md` and `gazebo_v10_comparison.csv`: generated
  paired summary.

Residual v10 used inference for 679/1,462 controller samples, applied a mean
0.095 m/s correction, and triggered 189 clearance-shield interventions. An
empty replan activated permanent fail-closed fallback, after which the
classical follower completed navigation and landing. This validates the
guarded hybrid pipeline, not uninterrupted direct residual transfer.

The v10 runs also validate the latest-value offboard transport fix. The former
unbounded FIFO consumed only one command per cycle and accumulated stale
setpoints because the MAVSDK consumer necessarily ran slower than its nominal
20 Hz producer. The fixed bridge drains queued commands and applies only the
newest velocity setpoint.

## Earlier diagnostic evidence

- `classical_smoke_v3.json`: classical navigation succeeded; landing was not
  yet included in the trial outcome.
- `residual_smoke_v3.json`: direct residual transfer timed out after repeated
  empty A* replans.
- `residual_smoke_v4.json`: an experimental straight-line recovery collided;
  that unsafe behavior was removed.
- `residual_failclosed_v5.json`: replacement fallback disabled residual
  control without collision, but the mission still timed out.

The residual integration remains disabled by default. It is bounded,
clearance-shielded, disable-able, and permanently fails closed to the classical
follower after an empty replan. More paired Gazebo trials are required before
making a reliability or performance claim.
