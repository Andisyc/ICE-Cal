# HP-4b r6 Order 1 Teacher Contract Block

Date: 2026-07-17

Status: `BLOCKED` at the learner teacher-checkpoint contract.

## Authorized scope

Execute the eight r6 A/B entries in frozen order, accepting each complete run
before starting the next. The frozen fail-fast contract requires the first
non-zero exit or acceptance failure to stop the sequence without retry,
override, source/config edit, or later-route execution.

## Attempted identity

- r6 identity SHA-256:
  `cbf054a84e9b44f4f6a104b8aa458821b5242bc3846731f266444ef88164b778`.
- Frozen cwd: `/private/tmp/unilab-hp4b-f66ab818`.
- Order 1: legacy repetition 1.
- Raw log:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r6/execution_logs/order_01_legacy.log`.
- Raw log SHA-256:
  `8f9a4a8817ee8042d2882f133fe03c9f659cad3adca648b70327b4731e56dc31`.

## Observed boundary

The r6 execution environment works: the command enters frozen Python/Hydra,
collects all three scenarios, writes their artifacts, and creates the cumulative
aggregate. The first failure is during `update_dagger_student()` before the
trainer or optimizer is constructed:

```text
ValueError: SAC teacher checkpoint obs dim mismatch:
checkpoint actor input dim=98 (net.0.weight),
configured teacher.obs_dim=99
```

Runtime path:

```text
run_multirole_dagger_workflow
-> update_dagger_student
-> run_offline_dataset_update
-> build_distillation_trainer
-> validate_sac_teacher_checkpoint_contract
```

## Owner audit

- Both workflow roles resolve task-native teacher specs with `obs_dim=98`.
- Both frozen teacher checkpoints have actor input dim 98.
- `conf/distill/workflow/g1_walk_stand.yaml` explicitly overrides the top-level
  student to `obs_dim=98` but does not override the top-level teacher.
- `conf/distill/config.yaml` defaults the top-level teacher to `obs_dim=99`.
- `update_dagger_student()` clones the composed root config; therefore the
  learner validates a 98-D role checkpoint against the unrelated 99-D generic
  default.

The first owner boundary is Hydra workflow configuration. A Python fallback or
silencing the checkpoint guard would violate UniLab's config-first and
fail-closed contracts.

## Partial artifact facts

- Manifest remains `BOOTSTRAP_COMPLETE` with zero completed DAgger iterations.
- Scenario datasets: 3 files, each 128 rows.
- Transition artifact: roles `walk_flat=32`, `stand=96`, age range `-1..23`.
- Aggregate: 1024 rows with scenarios `walk_flat=384`,
  `static_stand=384`, `walk_to_stop=256`.
- Aggregate SHA-256:
  `a9b35386a6fb7724d7606da32f50d4795a1e55ab9d084dd32c6fd911b9d97c40`.
- Metrics: 21 request records across the three scenarios; no workflow learner
  stages or cleanup-final record.
- Metrics SHA-256:
  `94dcc1aaeb5bb54fbb04789e74cb69eb8c8dab395829504a275db575f25e93cb`.
- No DAgger output checkpoint and no order-1 acceptance artifact.
- Orders 2-8 did not start.

These facts also runtime-confirm E51's repaired route: raw legacy role datasets
may omit scenario fields, while the explicit source identity annotation creates
a complete transition-aware aggregate before the learner boundary.

## Decision

HP-4b r6 is `BLOCKED` at order 1. No route comparison or speedup claim is
accepted. No retry, override, persistent run, or server mutation occurred.

The next bounded action requires separate authorization: workflow teacher
config-owner repair. Add the task-family-owned 98-D teacher contract through
the Hydra workflow configuration, prove compose/task-role agreement and the
real checkpoint preflight, refreeze source/config/executable identity, then
return before HP-4b.
