# oa_rl — Isaac Lab warehouse-avoidance research environment

Isaac Lab external project for AviationSim's quantitative comparison of a
classical obstacle-avoidance controller, standalone RL, and residual RL. The
registered task is `Isaac-WarehouseAvoidance-Direct-v0`.

The project now includes the residual architecture, a same-harness classical
baseline, deterministic and perturbed evaluation, compute instrumentation, an
identity-residual warm start, and nominal/moderate/severe result artifacts.

## Scope and architecture

The task reproduces the warehouse's 20 m × 14 m navigation area and three
staggered pillar rows. The drone is velocity-commanded in `(vx, vy)` while an
internal controller holds altitude at 1.5 m. The pillars are floor-to-ceiling,
so vertical avoidance is neither possible nor learned. Reaching the 2D goal
region is success; the verified Gazebo/PX4 ArUco landing FSM remains separate.

Three modes share the same scene, termination rules, and metrics:

- **Classical:** a 0.2 m-grid, 0.6 m-inflated Dijkstra goal-flow field, used as
  the vectorized Isaac analogue of repeated A* plus trajectory following.
- **Standalone RL:** the pre-residual 20-observation checkpoint directly
  commands planar velocity.
- **Residual RL:** the policy receives 22 observations, including the
  classical proposed velocity, and adds a correction scaled by `0.75 m/s` per
  axis. The combined command is capped at `1.5 m/s`.

Reward shaping uses progress in a bounded goal potential, not per-step goal
proximity. This fixed the earlier loitering exploit at 0.3015 m from a 0.3 m
goal radius.

## Setup

```bash
cd /home/berke/AviationSim/oa_rl
source ~/lab/bin/activate
python -m pip install -e source/oa_rl
```

## Main scripts

- `scripts/rsl_rl/train.py` — PPO training.
- `scripts/evaluate.py` — deterministic classical/standalone/residual
  evaluation; writes provenance-rich JSON and per-episode CSV. It requires
  `--episodes == --num_envs` and records exactly one first episode from each
  environment, sorted by scenario ID.
- `scripts/pretrain_zero_residual.py` — identity-residual warm start with an
  exact zero-output projection and PPO exploration std of 0.10.
- `scripts/compare_evaluations.py` — nominal three-way report.
- `scripts/compare_robustness.py` — success-conditioned robustness report.

The runner configuration uses the RSL-RL 5.x `actor`, `critic`,
`distribution_cfg`, and explicit `obs_groups` interface. The evaluator retains
legacy checkpoint conversion for the selected pre-5.0 policy.

## Nominal results

All rows use 1,024 deterministic episodes and one evaluator. Full report:
`results/evaluations/nominal_comparison.md`.

| Controller | Success | Time (s) | Path (m) | Efficiency | Command variation (m/s²) | Min clearance (m) | Residual (m/s) | Decision latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classical | 100% | 20.050 | 20.051 | 0.8329 | 8.648 | 0.335 | 0.000 | 0.146 |
| Standalone RL | 100% | 13.680 | 20.521 | 0.8138 | 9.423 | 0.075 | 0.000 | 0.200 |
| Penalized residual RL | 100% | 12.000 | 17.281 | 0.9664 | 1.167 | 0.375 | 0.522 | 0.328 |

Against classical, residual RL is 40.1% faster, follows a 13.8% shorter path,
improves efficiency by 16.0%, reduces command variation by 86.5%, and keeps
11.9% more minimum clearance. Its measured decision time is 0.328 ms, below
1% of the 40 ms control period.

## Identity-residual warm start and equal-budget finding

The first supervised checkpoint reached open-loop MSE `4.65e-5`, but retained
a `0.193 m/s` closed-loop correction. It timed out in all 1,024 episodes and
finished 14.58 m from the goal. This negative distribution-shift ablation is
`results/evaluations/2026-08-18_09-38-08_residual.json`.

The corrected pretrainer projects the actor head to the exact global optimum
for a zero-residual teacher. Deterministic output is zero while PPO exploration
std remains 0.10. Its 1,024-episode evaluation matched classical exactly:
`results/evaluations/2026-08-18_10-21-03_residual.json`.

Fine-tuning used 8,192 environments and 100 PPO iterations (19,660,800 steps,
52.67 s):

| Iteration | Initialization | Success | Time (s) | Efficiency | Residual (m/s) |
|---:|---|---:|---:|---:|---:|
| 50 | Identity warm start | 100% | 19.215 | 0.8480 | 0.046 |
| 50 | Random | 100% | 14.560 | 0.9157 | 0.298 |
| 100 | Identity warm start | 100% | 17.663 | 0.8674 | 0.109 |
| 100 | Random | 100% | 12.480 | 0.9543 | 0.526 |

The warm start is a safety/authority-control initialization, not a nominal
sample-efficiency win. Random initialization learns faster on the fixed
warehouse but uses substantially more residual authority.

## Held-out robustness evaluation

`evaluate.py --robustness_profile` supports:

- `nominal`: fixed spawn, unit actuator gain, no wind/noise.
- `moderate`: spawn jitter `(±0.25, ±0.75) m`, actuator gain `0.85–1.15`,
  wind up to `0.10 m/s`, observation-noise std `0.02`.
- `severe`: spawn jitter `(±0.50, ±1.25) m`, actuator gain `0.70–1.30`,
  wind up to `0.20 m/s`, observation-noise std `0.05`.

Separate seeded generators make first-episode scenario parameters and
observation noise reproducible without controller-dependent reset timing
shifting either stream. Shared observation features receive the same noise
even though the standalone policy has 20 inputs and residual/classical mode
has 22. Moderate results are complete; trajectory metrics are conditioned on
successful episodes so an early collision cannot look artificially efficient.

| Controller | Success | Collision | Successful time (s) | Successful efficiency | Reliability-adjusted efficiency | Clearance (m) |
|---|---:|---:|---:|---:|---:|---:|
| Classical | 100% | 0% | 20.531 | 0.8136 | 0.8136 | 0.367 |
| Standalone RL | 89.7% | 10.3% | 13.629 | 0.8169 | 0.7331 | 0.233 |
| Residual RL | 100% | 0% | 12.616 | 0.9445 | 0.9445 | 0.410 |

Full reports: `results/evaluations/nominal_comparison.md`,
`results/evaluations/moderate_robustness_comparison.md`, and
`results/evaluations/severe_robustness_comparison.md`.

The paired 1,024-scenario severe evaluation is also complete:

| Controller | Success | Collision | Successful time (s) | Successful efficiency | Reliability-adjusted efficiency | Clearance (m) |
|---|---:|---:|---:|---:|---:|---:|
| Classical | 60.1% | 0.0% | 20.870 | 0.8084 | 0.4855 | 0.375 |
| Standalone RL | 81.4% | 18.4% | 14.363 | 0.8183 | 0.6665 | 0.325 |
| Residual RL | 99.9% | 0.0% | 13.167 | 0.9423 | 0.9413 | 0.414 |

Residual RL completed 1,023 of 1,024 scenarios. It converted 408 of the
classical controller's 409 timeouts and 187 of standalone RL's 188 collisions
into successes. Its only failure was a bounded timeout, not a collision or
out-of-bounds termination. Full report:
`results/evaluations/severe_robustness_comparison.md`.

## Checkpoints and evidence

- Standalone final:
  `logs/rsl_rl/warehouse_avoidance_direct/2026-07-22_17-17-19/model_4999.pt`
- Penalized residual final:
  `logs/rsl_rl/warehouse_avoidance_direct/2026-07-23_13-05-22_residual-penalized/model_4999.pt`
- Corrected identity warm start:
  `logs/rsl_rl/warehouse_avoidance_direct/2026-08-18_10-19-25_zero-residual-il/model_il.pt`
- Identity-warm-start PPO:
  `logs/rsl_rl/warehouse_avoidance_direct/2026-08-18_10-22-35_il-residual-100/`
- Nominal source JSONs: `2026-08-18_09-06-27_classical.json`,
  `2026-08-18_08-43-10_standalone.json`, and
  `2026-08-18_08-44-40_residual.json` under `results/evaluations/`.
- Moderate source JSONs: `2026-08-18_11-28-22_classical_moderate.json`,
  `2026-08-18_11-29-21_standalone_moderate.json`, and
  `2026-08-18_11-30-32_residual_moderate.json`.
- Severe source JSONs are under
  `results/evaluations/severe_full_seed42/`: `2026-08-19_10-33-59_classical_severe.json`,
  `2026-08-19_10-35-22_standalone_severe.json`, and
  `2026-08-19_10-36-27_residual_severe.json`.

## Gazebo/PX4 transfer status

The final penalized residual checkpoint was exported to a dependency-light
NumPy actor and integrated as an opt-in, bounded correction around the ROS 2
classical follower. Earlier direct-transfer diagnostics exposed empty replans,
an unsafe recovery that was removed, and a collision-safe v5 fallback that
still timed out.

A separate offboard transport defect was then corrected: velocity setpoints now
use latest-value semantics instead of accumulating in an unbounded FIFO. In the
v10 equivalent-start smoke pair, both classical and residual-assisted runs
completed navigation and ArUco precision landing. Residual-assisted v10 took
74.18 s versus 76.52 s and followed 17.988 m versus 18.198 m, while minimum
clearance was 0.682 m versus 0.711 m. Residual inference covered 679/1,462
controller samples before an empty replan triggered permanent classical
fallback.

The wrapper remains bounded, clearance-shielded, disable-able, and fail-closed,
with no unplanned recovery motion. It remains disabled by default because the
successful Gazebo evidence is one pair and required fallback. The supported
claim is successful guarded hybrid transfer, not uninterrupted policy transfer
or statistical Gazebo superiority.

The consolidated report, plots, raw artifacts, and generated v10 comparison are
in `results/final_report.md`, `results/plots/`, and
`results/gazebo_transfer/`. No real-hardware validation was attempted.
