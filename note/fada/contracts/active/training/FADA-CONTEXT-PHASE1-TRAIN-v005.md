---
contract_id: FADA-CONTEXT-PHASE1-TRAIN-v005
status: active
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FADA-CONTEXT-PHASE1-TRAIN-v004
method_contract: FADA-CONTEXT-PHASE1-METHOD-v005
scope: fixed-left-knee-0.9 full-action SAC with forward-progress failure termination
---

# FADA Context Phase-1 Full-Action Training v005 Contract

## Runtime owner

`algo.runtime_impl=privileged_full_action_sac` and UniLab `DoubleBufferOffPolicyRunner` remain the
training owners. The only v005 runtime change is default-off G1 environment termination configured
by `env.forward_progress_termination`.

## Formal profile

Task owner: `sac/g1_walk_flat/mujoco_context_teacher_full_action_v005`.

The v004 compute profile, fixed command, fixed actuator strength, original-actor initialization,
trajectory rewards, and no-resume rule remain unchanged. The additional exact fields are:

```text
env.forward_progress_termination.enabled=true
env.forward_progress_termination.grace_steps=50
env.forward_progress_termination.min_command_forward_speed=0.1
env.forward_progress_termination.min_average_forward_speed=0.20
```

The first formal run starts from the original walking actor, not the rejected v004 checkpoint.

## Gates

Focused OFF/ON tests, fixed-yaw semantics, a real MuJoCo baseline/stationary discriminator, full
runtime regression, and CUDA no-training preflight must pass before launch. The formal evaluation
protocol and thresholds are unchanged from the v005 method contract.

## Forbidden behavior

- Do not merely increase reward weights in this retry.
- Do not resume or warm-start from the rejected v004 teacher checkpoint.
- Do not enable progress termination in unrelated G1 task owners.
- Do not classify time-limit truncation as progress failure.
- Do not accept training completion without paired quality.
