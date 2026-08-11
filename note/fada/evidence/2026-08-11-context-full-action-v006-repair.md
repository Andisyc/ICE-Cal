---
date: 2026-08-11
evidence_class: contract-confirmed
contracts: FADA-CONTEXT-PHASE1-METHOD-v006, FADA-CONTEXT-PHASE1-TRAIN-v006
status: training-completed-quality-failed
---

# Context Full-Action Teacher v006 Repair

## Failure boundary

v005 matched the original walking actor at initialization, but checkpoints 1000-5000 all failed at
the 50-step progress boundary. This localizes the first invalid boundary to actor optimization, not
checkpoint loading, keyboard command synchronization, or full-action initialization.

## Repair

The privileged teacher still emits one complete 29D action. Its SAC actor loss now includes a
training-only MSE anchor to the frozen original actor on the same observation. The anchor coefficient
is `10.0`; actor learning rate is `3e-5`; the run budget is 1000 iterations with saves every 100.

## Verification

- Focused command: `uv run pytest -q tests/algos/test_privileged_full_action_sac.py tests/algos/test_context_teacher_full_action_protocol.py tests/visualization/test_interactive_playback.py`.
- Result: `40 passed`, including a one-environment MuJoCo actor update.
- CPU no-training preflight: formal profile matched; dimensions were `(obs=98, critic=130, g=29,
  action=29)`; `full_action_output=true`; `residual_fusion=false`; anchor frozen; collector and
  training both remained stopped.
- Full `make test-all`: Ruff and mypy passed, pyright reported zero errors, and pytest completed with
  `1828 passed`, `50 skipped`, and `256 deselected`.

## Remaining boundary

Remote CUDA preflight passed with the same dimensions and frozen-anchor contract. Formal training
then completed `1000/1000` iterations and `2,070,528` environment steps in about 94 seconds. It
produced checkpoints every 100 iterations without traceback.

All ten checkpoints were screened with seed 101, 64 environments, and 60 steps. `model_100.pt`
restored walking: it survived 60/60 steps, advanced `0.347927 m`, and improved forward-velocity MAE
to `0.110040 m/s` versus the baseline's `0.129202 m/s`. Later checkpoints began degrading;
`model_900.pt` survived only 52.625 steps on average. This confirms that the action anchor repaired
the v005 stationary collapse but did not establish straight-line quality.

Formal five-seed, 256-environment, 400-step paired evaluation rejected both supported candidates:

| Metric | Baseline | model 100 | model 500 |
|---|---:|---:|---:|
| Forward progress | 2.580590 m | 2.819208 m | 2.519587 m |
| Forward-velocity MAE | 0.075800 m/s | 0.050228 m/s | 0.080509 m/s |
| Maximum lateral displacement | 0.122991 m | 0.180713 m | 0.360629 m |
| Maximum yaw drift | 0.142365 rad | 0.267305 rad | 0.205059 rad |
| Mean survival | 400 | 400 | 400 |
| Fall rate | 0.0 | 0.0 | 0.0 |

`model_100.pt` failed lateral and yaw reduction; `model_500.pt` additionally failed forward-velocity
non-degradation. Model 100 SHA-256 is
`cbc20c00da3688d5a35a7213468c2a05ab3df0855afc4340a3eead4c13a0c96a`; its formal report SHA-256 is
`e91e81a4c90dde14e312ea9381a1435c704e0871c7f5c5853d30ad725fbe969a`.

The runtime bug reported as "robot does not walk" is repaired, but no v006 checkpoint is an accepted
privileged teacher. Context Encoder training remains blocked. The next step is a human-owned method
decision about how to optimize trajectory direction without sacrificing the restored gait.
