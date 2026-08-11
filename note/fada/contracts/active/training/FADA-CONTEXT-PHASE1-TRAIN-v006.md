---
contract_id: FADA-CONTEXT-PHASE1-TRAIN-v006
status: active
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FADA-CONTEXT-PHASE1-TRAIN-v005
method_contract: FADA-CONTEXT-PHASE1-METHOD-v006
scope: behavior-anchored full-action SAC retry with early checkpoint discrimination
---

# FADA Context Phase-1 Full-Action Training v006 Contract

## Runtime owner

`algo.runtime_impl=privileged_full_action_sac` and UniLab `DoubleBufferOffPolicyRunner` remain the
owners. Task owner is `sac/g1_walk_flat/mujoco_context_teacher_full_action_v006`.

## Formal profile delta

All v005 physics, command, privilege, reward, termination, and paired-evaluation settings remain.
The exact v006 optimization fields are:

```text
algo.actor.nominal_action_anchor_coef=10.0
algo.actor_lr=0.00003
algo.max_iterations=1000
algo.save_interval=100
```

The run starts from the original walking checkpoint. It must not resume v004 or v005.

## Gates

Local owner tests, one-environment MuJoCo update, full regression, and remote CUDA no-training
preflight must pass before launch. After launch, checkpoints are screened in order at 100, 200, and
later iterations. Stop and reject the run at the first checkpoint whose fixed-0.9 rollout fails the
forward survival/progress discriminator; do not wait for iteration 1000 after confirmed collapse.

Only a checkpoint passing the unchanged formal paired gate is eligible to supervise Context Encoder.

## Forbidden behavior

- Do not treat the frozen anchor as a second deploy-time policy or action-residual branch.
- Do not optimize or serialize mutable anchor weights.
- Do not loosen the paired quality gate to accept nominal imitation alone.
- Do not start Context Encoder training from a merely finite or completed checkpoint.
