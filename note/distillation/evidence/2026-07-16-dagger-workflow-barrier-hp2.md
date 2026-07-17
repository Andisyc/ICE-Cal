# DAgger Workflow Barrier HP-2 Evidence

Date: 2026-07-16

## Scope

Connect the HP-1 persistent collector service to the DAgger scenario workflow
while preserving the outer iteration barrier and the legacy default path.
This is S1/S2 contract/connectivity evidence, not real G1 collection or a
training-speed claim.

## Core Parameter Trace

```text
current_checkpoint
-> scenario_collector.activate_checkpoint()
-> expected_weight_version
-> every DaggerCollectRequest/DaggerCollectResult in iteration k
-> scenario/iteration manifest evidence
-> update_student() creates student_(k+1)
-> next iteration activation
```

The two-iteration golden trace observed:

```text
activate bootstrap_student.pt version=41
collect walk_flat/static_stand/walk_to_stop version=41
update bootstrap_student.pt -> dagger_iteration_1.pt
activate dagger_iteration_1.pt version=42
collect walk_flat/static_stand/walk_to_stop version=42
update dagger_iteration_1.pt -> dagger_iteration_2.pt
```

## Migration Contract

- Owner switch: `execution_mode` in the workflow owner.
- OFF: `legacy`, existing `collect_scenario` callback, no persistent manifest
  fields.
- ON: `persistent_async`, `scenario_collector` required, legacy callback
  forbidden.
- Rejected mixed states: unknown mode, persistent mode without scenarios,
  persistent mode without service, both collectors supplied, or legacy mode
  with a persistent service.

## Red/Green Evidence

Before implementation, focused tests failed because
`run_multirole_dagger_workflow()` rejected the new `execution_mode` and
`scenario_collector` arguments.

Focused connector command:

```bash
uv run --active pytest tests/algos/test_distill_workflow.py -q \
  -k 'execution_mode or persistent_execution or connects_persistent or scenario_manifest_and_quota_sources'
```

Result: `4 passed, 12 deselected in 0.84s`.

The connector suite includes a real spawned
`PersistentDaggerCollectorRunner`, not only an in-process fake service. The
manifest records a worker PID different from the parent and the same version
for both scenario artifacts.

Final affected gate:

```text
345 passed, 5 warnings in 11.14s
```

Ruff and atlas validation are recorded in the final HP-2 gate. The atlas check
reports `runtime_modules=9 method_modules=11 concept_nodes=6`.

## Stale Search

The repository search found the new mode and weight-version fields only in the
workflow/runtime owners, tests, and governance notes. `scripts/train_distill.py`
still supplies the legacy `collect_scenario` callback and does not select
`persistent_async`; this is the explicit HP-3 connector gap, not a hidden ON
path.

## Decision

HP-2 passes. The workflow can now use the HP-1 persistent runner through an
explicit ON route, and the legacy default retains its old callback and manifest
shape. HP-3 may add the Hydra/script connector and real distillation runtime
factory.

## Unconfirmed

- real G1 env/teacher/student persistence;
- real `SharedWeightSync` checkpoint publication;
- measured stage throughput or speedup;
- Motrix backend support;
- policy or physical quality.
