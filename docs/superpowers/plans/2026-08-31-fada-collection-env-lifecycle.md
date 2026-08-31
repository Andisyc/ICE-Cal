# FADA Collection Environment Lifecycle Repair

## Accepted behavior

Each call that collects one FADA source batch owns exactly one G1WalkFlat environment and its
MuJoCo native pool. The environment is created before that collection transaction and closed in a
`finally` boundary immediately after it. Standing and walk-to-stand remain command scenarios of
the same G1WalkFlat task.

## Preserved behavior

- Privileged Oracle, student, 20+1 checkpoint lineage, source allocations, window semantics, and
  optimizer schedule are unchanged.
- Observation, action, Reward, domain-randomization, checkpoint, and artifact contracts are
  unchanged.
- The spawned Collector process and resident policy modules remain persistent; only the native
  environment lifetime is shortened.

## Owner and lifecycle

`PersistentFADACollectorWorker` owns environment construction, Oracle-environment validation,
physics-guard configuration, cumulative diagnostics, and closure. `collect_fada_iteration` enters
that owner boundary once for every main/profile/intermediate source transaction and never stores
or reuses an environment itself.

## Proof route

1. A regression must fail while `collect_fada_iteration` bypasses the lifecycle boundary.
2. Constructor tests must prove successive transactions receive distinct environments, close each
   exactly once, and retain one-task standing semantics.
3. Focused FADA worker/window tests, the affected algorithm suite, Ruff, compileall, and
   `git diff --check` must pass.
4. Offline evidence cannot prove remote MuJoCo native stability; one bounded server run remains the
   final confirmation.
