---
date: 2026-08-11
evidence_class: runtime-confirmed
contracts: FADA-CONTEXT-PHASE1-METHOD-v004, FADA-CONTEXT-PHASE1-TRAIN-v004
status: quality-failed
---

# Context Full-Action Teacher v004 Evaluation

## Training completion

The remote process exited normally with `status=completed`, `5000/5000` updates, and `10,262,528`
environment steps. Training wall time was about `8m53s`. The final checkpoint is:

```text
/ssd1/cyx/FADA_runs/20260811_context_teacher_full_action_v004/training/model_5000.pt
sha256=df9f4ea39a986dfa53d3799df271def2f1b9b8985b538518cfd593a459a19bcf
```

The checkpoint contains schema `unilab_privileged_full_action_teacher_v1`, dimensions
`(obs=98, g=29, action=29)`, update count `5000`, and the accepted nominal initialization SHA-256.

## Formal paired evaluation

The exact same-snapshot protocol used held-out seeds `[101, 102, 103, 104, 105]`, `256`
environments per seed, `400` steps, fixed command `(0.4, 0.0, 0.0)`, and fixed left-knee strength
`0.9` for both policies.

| Metric | Original policy at 0.9 | Full-action teacher at 0.9 | Result |
|---|---:|---:|---|
| Max lateral displacement | `0.122991 m` | `0.006117 m` | `95.03%` lower |
| Max yaw drift | `0.142365 rad` | `0.048456 rad` | `65.96%` lower |
| Forward-velocity MAE | `0.075800 m/s` | `0.402657 m/s` | failed non-degradation |
| Forward progress | `2.580590 m` | `-0.025603 m` | teacher did not walk forward |
| Lateral-velocity MAE | `0.036735 m/s` | `0.011239 m/s` | lower |
| Fall rate | `0.0` | `0.0` | passed |
| Action saturation step rate | `0.0` | `0.0` | passed |

Every seed produced the same qualitative result. The teacher stayed stable but moved slightly
backward instead of tracking the commanded `0.4 m/s`. The quality gate failed only
`forward_velocity_non_degradation`; protocol pairing, seed identity, lateral reduction, yaw
reduction, fall rate, and saturation all passed.

Artifact:

```text
/ssd1/cyx/FADA_runs/20260811_context_teacher_full_action_v004/evaluation/model_5000_full_action_paired.json
```

## First invalid behavior boundary

The result is reward exploitation, not successful straight walking. The composed reward has
`alive=10`, `tracking_lin_vel=2`, `penalty_lateral_displacement=-20`, and
`penalty_yaw_drift=-10`. Runtime evidence shows the learned policy can obtain low lateral/yaw error
by remaining nearly stationary. This checkpoint is not eligible to supervise Context Encoder.

Do not tune and retrain automatically. The next method decision is whether to enforce forward
motion through a stronger command-tracking/progress owner, a rejection constraint, or a different
teacher-training formulation.
