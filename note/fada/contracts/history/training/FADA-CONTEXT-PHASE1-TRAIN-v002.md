---
contract_id: FADA-CONTEXT-PHASE1-TRAIN-v002
status: superseded
effective_date: 2026-08-10
updated_date: 2026-08-10
supersedes: FADA-CONTEXT-PHASE1-TRAIN-v001
method_contract: FADA-CONTEXT-PHASE1-METHOD-v002
scope: training-ready default-off privileged residual SAC teacher route with formal paired acceptance
---

# FADA Context Phase-1 Training Contract

## Runtime owner

`algo.runtime_impl=privileged_residual_sac` is the single owner selection for this route. Existing SAC
configs that do not select this runtime preserve their actor, observation dimensions, collection,
checkpoint, and playback behavior.

The route uses UniLab `DoubleBufferOffPolicyRunner` with its collector process, shared packed replay,
CPU-pinned double-buffer prefetch, learner, and synchronized actor-weight publication. Collection is
barrier-synchronized at one environment step per sync; this is not the Planner-IDM
`persistent_async` distillation runtime and not a new synchronization protocol.

The Phase-1 route loads `checkpoints/oracles/G1WalkFlat/model_5000.pt`, SHA-256
`db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`, as a frozen nominal actor and
trains only the privileged residual branch. The SAC critics consume the fused executed action.
Current and next `g` values come from the final 29 dimensions of critic observations stored in replay;
the pre-existing critic-only linear velocity remains separate.

## Formal training profile

The task owner is `sac/g1_walk_flat/mujoco_context_teacher_phase1`. Its initial formal profile fixes:

```text
num_envs=2048                 batch_size=8192
replay_buffer_n=512           updates_per_step=8
learning_starts=10            policy_frequency=4
env_steps_per_sync=1          max_iterations=5000
save_interval=1000            use_symmetry=false
```

The environment samples one 29D `g` per reset and applies the same row to Kp, Kd, privileged info,
and critic observation. Only knee candidates `[3, 9]` are used, multiplier range is `[0.85, 0.95]`,
nominal probability is `0.2`, unrelated Kp/Kd randomization is disabled, and the command is fixed to
`(0.4, 0.0, 0.0)` m/s. Automatic post-training playback is disabled for the formal run.

The launch must declare `training.device` and `training.log_dir` explicitly. Resume/warm-start beyond
the frozen nominal checkpoint is forbidden for the first formal run. Changing the formal profile
requires a new contract version rather than an ad hoc Hydra override.

## Preflight and acceptance

Before launch, the preflight owner must validate the composed task, runtime/runner/learner types,
actor/critic/action dimensions `(98, 130, 29)`, privileged tail `29`, nominal checkpoint identity,
formal hyperparameters, strength distribution, fixed command, and no-play setting. Preflight must not
call `runner.learn`, start a collector, create a run directory, or save a checkpoint.

Formal evaluation uses the exact protocol and conjunctive thresholds in
`FADA-CONTEXT-PHASE1-METHOD-v002`. A checkpoint evaluated with different seeds, environment count,
horizon, command, or strength profile remains `unassessed`. Formal training completion is not teacher
quality acceptance; only a checkpoint with machine-readable `quality_status=passed` is eligible for
the later Context Encoder stage.

## Forbidden behavior

- Do not update or replace the nominal SAC actor.
- Do not expose `g` to the actor observation or future Context Encoder.
- Do not use a fixed single left-knee case as formal training evidence.
- Do not bypass the UniLab runner lifecycle or introduce a second collector/learner protocol.
- Do not launch training without explicit user authorization for the target device and log path.
- Do not claim teacher or Context quality from startup, finite loss, survival alone, or a shortened
  evaluation sentinel.
