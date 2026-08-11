---
date: 2026-08-11
evidence_class: runtime-confirmed
contracts: FADA-CONTEXT-PHASE1-METHOD-v005, FADA-CONTEXT-PHASE1-TRAIN-v005
status: training-started
---

# Context Full-Action Teacher v005 Launch

v005 preserves the independent privileged full-action teacher and fixed left-knee `0.9` experiment.
It does not resume the rejected v004 teacher and does not alter reward weights, SAC ownership, or the
paired acceptance gate. Its only behavioral change is a default-off G1 episode failure condition:
after 50 steps, commanded-forward rows must have at least `0.20 m/s` average forward speed in the
reset-yaw frame.

## Boundary calibration and discriminator

- Original checkpoint under fixed left-knee `0.9`, seeds 101-105, 256 environments: minimum
  step-50 average forward speed was `0.226277 m/s`, leaving about 11.6% margin.
- Remote real-MuJoCo discriminator, seed 101, 64 environments, 60 requested steps: original policy
  survived 60 steps with fall rate `0`; rejected v004 stationary teacher survived exactly 50 steps
  with fall rate `1.0`.
- The diagnostic JSON is
  `/ssd1/cyx/FADA_runs/20260811_context_teacher_full_action_v005/diagnostics/v004_stationary_discriminator.json`.

## Verification

- Local focused plus wider regression: `83 passed`; the single PyTorch shared-memory case blocked
  by the macOS sandbox passed when rerun with shared-memory permission.
- Local no-training preflight: passed with task
  `sac/g1_walk_flat/mujoco_context_teacher_full_action_v005`.
- Remote Python 3.10 focused suite: `39 passed`.
- Remote CUDA no-training preflight: `status=passed`, `training_started=false`,
  `collector_started=false`, dimensions `(obs=98, critic=130, g=29, action=29)`.
- Preflight recorded `full_action_output=true`, `residual_fusion=false`, fixed strength index
  `3 = 0.9`, and nominal checkpoint SHA-256
  `db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`.

## Formal launch

Remote run root:

```text
/ssd1/cyx/FADA_runs/20260811_context_teacher_full_action_v005
```

The formal process launched on GPU 0 with PID `117492`. One startup check confirmed the process was
alive and the log contained:

```text
[DoubleBufferRunner] Collection sync enabled: env_steps_per_sync=1
[DoubleBufferRunner] Collector process alive: True
[Collector] Entry point called
```

Monitoring stopped immediately after this startup confirmation, following the human's SSH training
instruction. Training completion, checkpoint existence, and paired quality remain unverified.
