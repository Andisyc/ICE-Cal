# DAgger Collector Metrics Connector HP-4a2b Evidence

Date: 2026-07-16

Scope: opt-in collector-owned teacher/student inference, env-step, and
tensor-pack observations; copied dataset metadata; validated persistent-worker
pass-through. No reset/resource timing, parent identity, JSON artifact, live
MuJoCo, training, A/B, or optimization is included.

## Owner Boundary

The collector measures only code it owns:

- `teacher_inference`: all input rows actually forwarded through teacher actors;
- `student_inference`: all rows forwarded through rollout actors;
- `env_step`: direct `env.step` calls, excluding done/reset repair;
- `tensor_pack`: selected-row conversion/chunk assembly plus final dataset build.

Cached request reset occurs in `PersistentResourceCache`, so HP-4a2b does not
mislabel it as collector time. `performance_clock=None` is the default and
emits no performance metadata, preserving the legacy dataset path.

## Core Parameter Trace

```text
injected clock
-> DistillationStageObservationAccumulator per-stage sum
-> exact inference-row / env-step / packed-row counts
-> DistillationStageObservation validation
-> dataclasses.replace(dataset, copied metadata)
-> worker schema/order validation
-> DaggerCollectResult.metadata
```

## Tiny Golden Facts

Role collector, two envs and four accepted rows:

```text
stage order = teacher_inference, student_inference, env_step, tensor_pack
durations = 2, 2, 1, 3 seconds
row counts = 4, 4, 0, 4
env-step counts = 0, 0, 1, 0
```

Transition collector, two envs and eight rows:

```text
durations = 4, 4, 3, 5 seconds
row counts = 16, 8, 0, 8
env-step counts = 0, 0, 3, 0
```

The 16 teacher rows are intentional: both walking and standing teachers run on
each transition state before row-wise target selection. With one rollout
policy, student inference is eight rows; the command-intent two-expert path
will count both executed forwards.

## Verification

Test-first red: role/transition collectors rejected `performance_clock`, and
the worker returned only its three HP-4a2a stages.

Fresh evidence:

- focused collector/schema/worker: `23 passed in 0.07s`;
- complete distillation collector/data contract: `80 passed, 5 warnings`;
- persistent runtime/workflow impact: `33 passed in 4.00s`;
- Ruff after mechanical import sorting: pass.

The semantic fixtures cross actual role/transition collector loops. Worker
pass-through uses a schema-valid semantic payload but is not a live MuJoCo
timing claim.

## Decision

HP-4a2b passes its S1/S2 boundary. HP-4 runtime instrumentation remains
partial: resource/reset timing, parent identity enrichment, run-local
persistence, cleanup-final records, and live comparison remain absent.
HP-4a2c, Gate 0B, and HP-4b remain blocked pending separate user decisions.
