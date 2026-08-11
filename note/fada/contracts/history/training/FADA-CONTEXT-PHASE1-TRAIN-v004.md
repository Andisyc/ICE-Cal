---
contract_id: FADA-CONTEXT-PHASE1-TRAIN-v004
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FADA-CONTEXT-PHASE1-TRAIN-v003
superseded_by: FADA-CONTEXT-PHASE1-TRAIN-v005
method_contract: FADA-CONTEXT-PHASE1-METHOD-v004
scope: fixed-left-knee-0.9 UniLab privileged full-action SAC training and paired evaluation
---

# FADA Context Phase-1 Full-Action Training Contract

## Runtime owner

`algo.runtime_impl=privileged_full_action_sac` selects the full-action teacher. The route reuses
UniLab `DoubleBufferOffPolicyRunner`; it does not create a new collector protocol. Existing SAC and
the historical `privileged_residual_sac` runtime remain unchanged when this owner is not selected.

The teacher actor consumes `98D` actor observation and the final `29D` motor-strength tail from the
`130D` critic observation, then directly outputs `29D` action. The original walking checkpoint is
used only for actor initialization and paired baseline evaluation.

## Formal profile

The task owner is `sac/g1_walk_flat/mujoco_context_teacher_full_action_left_knee_090`. The initial
formal compute profile retains the v003 SAC budget for a controlled comparison:

```text
num_envs=2048                 batch_size=8192
replay_buffer_n=512           updates_per_step=8
learning_starts=10            policy_frequency=4
env_steps_per_sync=1          max_iterations=5000
save_interval=1000            use_symmetry=false
```

All environments use a fixed 29D multiplier vector with only index `3` set to `0.9`. The command is
fixed to `(0.4, 0.0, 0.0)` m/s. Resume is forbidden for the first run. Training device and a new log
directory must be explicit.

## Gates

Before formal training, focused tests and a no-training preflight must prove runtime selection,
dimensions, exact strength vector, nominal warm-start identity, full-action checkpoint metadata,
and collector privileged-input connectivity. A bounded one-environment MuJoCo update may establish
live connectivity but cannot establish quality.

After training, only the exact paired protocol and thresholds in
`FADA-CONTEXT-PHASE1-METHOD-v004` may accept the checkpoint for future Context supervision.

## Forbidden behavior

- Do not compute `nominal_action + delta_action` in the v004 teacher.
- Do not train on nominal or randomized strength rows in the formal v004 run.
- Do not compare teacher-at-`0.9` against original-policy-at-`1.0`.
- Do not pass `g` into the baseline actor.
- Do not launch formal training before preflight and explicit launch handoff.
