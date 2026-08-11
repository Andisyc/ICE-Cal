---
date: 2026-08-11
evidence_class: runtime-confirmed
contracts: FADA-CONTEXT-PHASE1-METHOD-v004, FADA-CONTEXT-PHASE1-TRAIN-v004
status: training-started
---

# Context Full-Action Teacher v004 Launch

The v004 retry replaces the historical residual teacher with an independent privileged policy:

```text
teacher(actor_observation, 29D actuator_strength) -> complete 29D action
```

The original walking checkpoint is used only to initialize the teacher and to provide the paired
baseline. It is not present as a frozen branch in the teacher forward pass.

## Verification

- Local focused suite: `84 passed`; one unrelated shared-memory test was permission-blocked in the
  sandbox and passed when rerun with shared-memory permission.
- Local one-environment MuJoCo step plus critic/actor update: `1 passed`, finite action and losses.
- Remote focused suite under Python 3.10/CUDA environment: `26 passed`.
- Remote CUDA no-training preflight: `status=passed`, `training_started=false`,
  `collector_started=false`, dimensions `(obs=98, critic=130, g=29, action=29)`.
- Preflight recorded `full_action_output=true`, `residual_fusion=false`, actor parameter count
  `245930`, and nominal initialization SHA-256
  `db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`.
- Fixed strength is exactly left-knee index `3 = 0.9`, with every other entry `1.0`.

## Formal launch

Remote run root:

```text
/ssd1/cyx/FADA_runs/20260811_context_teacher_full_action_v004
```

The formal process launched on GPU 0 with PID `114506`. One startup check confirmed the process was
alive and the log contained:

```text
[DoubleBufferRunner] Collection sync enabled: env_steps_per_sync=1
[DoubleBufferRunner] Collector process alive: True
[Collector] Entry point called
```

Monitoring stopped immediately after startup confirmation, following the human's standing SSH
training instruction. Training completion, checkpoint existence, and paired quality remain
unverified until the human requests the next check.
