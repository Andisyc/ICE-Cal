# DAgger Production Persistent Runtime HP-3b2 Evidence

Date: 2026-07-16

Scope: OFF-default production factory wiring, exact-identity resident G1
teacher/env resources, per-request reset isolation, existing collector/data
semantics, and a bounded local MuJoCo lifecycle sentinel. This evidence does
not authorize training, claim physical policy quality, cover Motrix, or modify
the active server run.

## Owner Route

```text
Hydra execution_mode=persistent_async
-> scripts/train_distill.py assembly
-> PersistentDistillationRuntime + UniLab AsyncRunner/SharedWeightSync
-> PersistentG1DistillationWorker
-> PersistentResourceCache(exact owner identity)
-> existing role/transition collectors
-> existing DistillationTensorDataset persistence
```

The default remains `execution_mode=legacy`. The worker caches resources by
task owner, task name, backend, resolved env config fingerprint, `num_envs`,
teacher checkpoint path/hash, and teacher spec fingerprint. Role strings are
not cache keys.

## Contract And Differential Evidence

- exact-key resource lifecycle: `2 passed`;
- legacy/persistent collector differential: `3 passed`;
- production worker sequence fixture: `1 passed`;
- persistent runtime lifecycle: `2 passed`;
- fail-closed sentinel summary validation: `2 passed`;
- production factory entrypoint fixtures: `2 passed, 195 deselected`;
- affected Ruff check: `All checks passed!`.

Final impact groups: IPC/runtime `59 passed in 5.91s`; distillation,
workflow, config, and script `419 passed, 5 warnings in 17.36s`. Atlas check:
`runtime_modules=9 method_modules=11 concept_nodes=6`.

The differential compares sample dimensions, role labels, command intents,
scenario labels, transition ages, cached teacher actions, teacher identity,
and one-reset-per-request behavior through the actual collector/data owners.

## Bounded Real MuJoCo Sentinel

One local spawned worker (PID `62266`) served
`walk_flat -> static_stand -> walk_to_stop -> walk_flat`, with one env and four
rows per request. All four results observed student weight version `1`.

Final close report:

```text
student_init_count=1
teacher_init_count=2
env_init_count=2
request_count=4
reset_count=4
cache_hit_count=3
request_error_count=0
teacher_close_count=2
env_close_count=2
```

The transition rows preserve intents
`active, active, inactive, inactive`, roles
`walk_flat, walk_flat, stand, stand`, and ages `-1, -1, 0, 1`.
The two exact cache keys were reused without additional teacher/env creation.
Only Gymnasium Box cast-overflow warnings were emitted. The sentinel now
fails closed on scenario order, worker/version identity, transition semantics,
missing close reports, resource counts, or incomplete cleanup.

Teacher checkpoint SHA-256 identities:

- walking: `7a0729a45859b2db05f2a642f6e80eedbd25f8135a75ff2af9dddae58bbf8279`;
- standing: `91e18d3d1f469b2bead350cd41b33494c39c8ec8d26f2daf802e0273afa2c6da`.

## Decision

HP-3b2 passes the production connectivity and lifecycle gate. This is the
mandatory stop boundary: no bounded persistent training starts until the user
explicitly authorizes it. Persisted stage timing/throughput and physical
checkpoint acceptance remain separate gates.
