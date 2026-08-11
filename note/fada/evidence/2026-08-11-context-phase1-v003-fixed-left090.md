# Context Phase-1 v003 Fixed-Left-Knee Evidence

Date: 2026-08-11
Contracts: `FADA-CONTEXT-PHASE1-METHOD-v003`, `FADA-CONTEXT-PHASE1-TRAIN-v003`
Remote run: `/ssd1/cyx/FADA_runs/20260811_context_teacher_phase1_v003_fixed_left090`

## Implemented scope

The active reset distribution contains only nominal `g=1.0` and left-knee index `3` fixed at
`0.9`, with probability `0.5` per stratum. The task adds initial-yaw-frame squared lateral-
displacement and yaw-drift penalties. Formal evaluation rejects right-knee and non-`0.9` anomaly
rows.

## Verification and preflight

- Local owner suite: `28 passed`.
- Local runtime/collector/playback suite: `136 passed`; the one macOS shared-memory case that failed
  inside the restricted sandbox passed when rerun with shared-memory permission.
- Local 64-environment MuJoCo probe emitted only nominal and left-knee-0.9 rows (`40/24` in that
  seeded process), retained episode-frame state, and produced finite one-step rewards.
- Remote owner suite: `28 passed` under Python 3.10.
- CUDA preflight: `status=passed`, `training_started=false`, `collector_started=false`, dimensions
  `(obs=98, critic=130, action=29, g=29)`, and exact nominal checkpoint SHA-256
  `db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`.

## Formal training

The UniLab `DoubleBufferOffPolicyRunner` completed `5000/5000` iterations and `10,262,528`
environment steps in about eight minutes. The run summary reports `status=completed`, no traceback,
and final mean episode length `1000`. The final checkpoint is:

```text
/ssd1/cyx/FADA_runs/20260811_context_teacher_phase1_v003_fixed_left090/training/model_5000.pt
sha256=c1ecbfb81f170e7ffdea96b38ae2cb432b23f89eca8640e8a18d747a081886da
```

Training completion is runtime evidence only. TensorBoard showed the lateral-displacement penalty
changing from about `-0.0014` to `-22.86` and yaw-drift penalty from about `-0.0081` to `-2.16`, so
the checkpoint required formal rejection testing rather than acceptance from survival.

## Formal paired evaluation

The exact five-seed, 256-environment, 400-step protocol matched with no mismatches and covered `656`
nominal plus `624` left-knee rows. The result was `quality_status=failed`.

| Scenario | Branch | Max lateral (m) | Max yaw (rad) | Forward MAE (m/s) | Fall rate |
|---|---|---:|---:|---:|---:|
| nominal | frozen nominal | 0.142158 | 0.194307 | 0.078837 | 0.0 |
| nominal | teacher | 1.219089 | 0.927236 | 0.072116 | 0.0 |
| left knee 0.9 | frozen nominal | 0.119491 | 0.142461 | 0.076080 | 0.0 |
| left knee 0.9 | teacher | 1.242342 | 0.945678 | 0.067769 | 0.0 |

Failed checks were left-knee maximum lateral/yaw reduction plus nominal final/maximum lateral,
final/maximum yaw, and lateral-velocity non-degradation. The teacher residual L2 mean was about
`0.472` in both strata, with zero clipping and zero falls. Similar outcomes in both strata are not
proof that the actor ignores `g`; an explicit matched-observation `g`-sensitivity probe is the
smallest discriminator.

Artifact:

```text
/ssd1/cyx/FADA_runs/20260811_context_teacher_phase1_v003_fixed_left090/evaluation/model_5000_formal_paired.json
```

## Remaining boundary

The v003 checkpoint is not eligible for Context Encoder training. Do not retrain from reward tuning
alone until a runtime probe separates three hypotheses: residual branch ignores `g`, reward/Q scale
causes an optimization failure, or the learned residual reacts to state in a direction inconsistent
with the initial-frame precision objective.

## Frozen-nominal strength scan

A follow-up probe disabled the teacher residual and evaluated the exact frozen walking actor for
left-knee gain multipliers `1.0`, `0.9`, `0.8`, and `0.7`. It used the same five seeds, 256
environments per seed, 400-step horizon, and fixed `0.4 m/s` command.

| Left-knee multiplier | Max lateral (m) | Max yaw (rad) | Forward MAE (m/s) | Fall rate |
|---:|---:|---:|---:|---:|
| 1.0 | 0.141951 | 0.195811 | 0.078622 | 0.0 |
| 0.9 | 0.122991 | 0.142365 | 0.075800 | 0.0 |
| 0.8 | 0.094300 | 0.076127 | 0.073721 | 0.0 |
| 0.7 | 0.126660 | 0.143617 | 0.076995 | 0.0 |

Every per-seed result followed the same pattern; aggregation did not hide an adverse seed. This
runtime evidence falsifies the assumption that Kp/Kd gain scaling in `[0.7, 0.9]` creates a useful
execution fault for the frozen walking policy. It does not establish whether lower gains, torque
limits, latency, or external wrench disturbances create a valid controlled anomaly.

Artifact:

```text
/ssd1/cyx/FADA_runs/20260811_context_teacher_phase1_v003_fixed_left090/diagnostics/nominal_left_knee_strength_scan.json
```
