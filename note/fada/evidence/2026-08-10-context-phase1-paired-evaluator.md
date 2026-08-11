# Context Phase-1 Paired Evaluator Evidence

Date: 2026-08-10

Contracts: `FADA-CONTEXT-PHASE1-METHOD-v001`, `FADA-CONTEXT-PHASE1-TRAIN-v001`

## Implemented boundary

- `scripts/evaluate_context_teacher_phase1.py` is an explicit, default-off evaluation owner.
- It reconstructs the privileged residual actor through the Phase-1 runtime and strict-loads the
  teacher checkpoint, including nominal checkpoint hash/tensor identity validation.
- Nominal and residual branches start from one captured environment snapshot. The evaluator verifies
  exact equality of actor observation, command, 29D `g`, base pose, and local linear velocity after
  restore; the snapshot also owns physics, environment carrier, counters, pending forces, RNG, and
  autoreset state.
- Autoreset is disabled in both branches. Each row stops accumulating after its first termination or
  truncation, and those events are reported separately.
- Displacement is rotated into each row's initial yaw frame. Metrics serialize overall and by
  nominal, left-knee, and right-knee strata. Residual and clipping values remain diagnostics and are
  excluded from the lower-is-better trajectory-improvement group.
- Rows outside the Phase-1 structural profile (nominal or exactly one changed knee actuator) fail
  closed. The configured knee strength remains the declared continuous range `[0.85, 0.95]`.

## Real MuJoCo sentinel

Command boundary: 64 environments, 100 steps, seed 11, CPU, fixed command `(0.4, 0.0, 0.0)`.

Teacher checkpoint:

```text
/private/tmp/fada-context-phase1-sentinel2-20260810/model_1.pt
sha256=ef26e7cfdeb20536b8c17a62e6e1526077ac15d0b22ff7dd8bee5f10988571b2
update_count=1
```

Structured report:

```text
/private/tmp/fada-context-phase1-paired-sentinel2-20260810.json
sha256=ce0d0b48c70da75d24ec47770114c4d22af65d7683cd555073364d5427ec94a4
scenario_counts: nominal=9, left_knee=22, right_knee=33
pairing_exact_for_all_seeds=true
```

Selected overall measurements:

| Metric | Nominal | One-update teacher | Lower-is-better improvement |
|---|---:|---:|---:|
| Forward velocity MAE (m/s) | 0.106846 | 0.104651 | +0.002195 |
| Final lateral displacement (m) | 0.046658 | 0.046217 | +0.000441 |
| Maximum lateral displacement (m) | 0.062672 | 0.063860 | -0.001188 |
| Final yaw drift (rad) | 0.047830 | 0.028490 | +0.019340 |
| Maximum yaw drift (rad) | 0.072625 | 0.058677 | +0.013947 |
| Fall rate | 0.0 | 0.0 | 0.0 |
| Truncation rate | 0.0 | 0.0 | 0.0 |

Teacher diagnostics were residual L2 mean `0.014861`, residual L-infinity maximum `0.003007`, and
zero element/step clipping. The mixed trajectory result is expected from a one-update engineering
sentinel and does not establish teacher quality.

## Verification

- Paired evaluator semantic suite: `7 passed`.
- Privileged actor, actuator-strength, and evaluator focused suite: `21 passed`.
- Relevant G1, `NpEnv`, off-policy, HORA, playback, and evaluator regression: `331 passed`. Four
  shared-memory/socket tests were permission-blocked inside the sandbox and passed when rerun with
  those capabilities.
- Ruff: passed.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- `git diff --check`: passed.

Gymnasium emitted the existing float-bound cast warnings during MuJoCo construction. No tested state,
metric, action, or checkpoint identity became non-finite.

## Not established

No numeric quality thresholds have been accepted, no formal teacher training has been launched, and
this sentinel does not establish Context Encoder identifiability, real-robot correction, or novelty.
