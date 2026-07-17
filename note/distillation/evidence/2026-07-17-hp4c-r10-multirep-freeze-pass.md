# HP-4c r10 Multi-Repetition Two-Iteration Freeze Pass

Date: 2026-07-17

Status: `PASS` for the freeze-only r10 benchmark gate. No benchmark run or
training artifact exists.

## Frozen identity

- Freeze directory:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_multirep_20260717_r10`.
- Benchmark identity SHA-256:
  `8f14c14c3e8993c2d943a4871347d0d0bf33675194595946f7e0f1fd9a3f1185`.
- Oracle v4 SHA-256:
  `9acbbef280203ad1b3fce686a01dbea9aeb77462cc6a765d50405f75d09683a0`.
- Oracle contract SHA-256:
  `9329719d4dcbac6f0e83170faad8a2f2037a2b1777c885c28ca6248d34a6aed6`.
- Frozen preflight SHA-256:
  `4d97819a7905d68ef985b325ed91aaed7ee736719d478dcbfc1b61f254801fb4`.
- Formal output root:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_multirep_ab_20260717_r10`.
- Freeze state: output root absent, execution logs absent,
  `execution_authorized=false`, and `training_started=false`.

## Workload and order

The workload is byte/semantic identity with r9 except for repetition count:

- two DAgger outer iterations;
- CPU, seed 1, four collection envs;
- 128 rows per scenario in `walk_flat`, `static_stand`, `walk_to_stop` order;
- quotas `0.50, 0.25, 0.25`;
- configured update floor 8, with accepted actual counts `16 -> 24`;
- four repetitions per route, eight total runs.

The balanced execution order is frozen as:

`L1, P1, P2, L2, L3, P3, P4, L4`.

Each order owns a distinct run directory and full Hydra compose SHA-256. All
eight configs become identical after removing only
`training.workflow.execution_mode` and `training.workflow.run_dir`; the shared
hash is `25c2bb69...0055`.

## Pre-registered decision rule

Primary metric: complete `subprocess_elapsed_seconds`. Pair by repetition and
compute `persistent_async / legacy`.

A stable direction speedup may be recorded only when all are true:

- median paired ratio is below 1;
- persistent median is below legacy median;
- at least three of four paired ratios are below 1.

Otherwise the frozen verdict is `NO_STABLE_SPEEDUP`. Four pairs are a
deterministic direction gate, not inferential significance. Request/cache/
cleanup/workflow metrics are secondary mechanism evidence and cannot override
the primary process metric. Policy quality and physical acceptance are outside
this benchmark.

## No-training preflight

The preflight verifies:

- exact r8 source bundle `ea1d4f7a...b25e` and all 1252 source files;
- inherited exact-source suite `493 passed` and Ruff evidence hashes;
- all assets and two real 98-D teacher checkpoint contracts;
- MuJoCo `nq/nv/nu=36/35/29`;
- all eight per-order compose hashes and the shared hash;
- oracle syntax and hash;
- nested frozen-source import/help;
- normalized 171-entry dependency provider identity;
- r9 identity/execution/verdict derivation hashes;
- output root and every run directory absent.

## Decision

r10 is frozen and ready for a separately authorized eight-run execution. This
step does not authorize order 1. A future executor must use the exact frozen
order and invoke oracle v4 after every successful run, stopping at the first
train or oracle failure. HP-5 and default-on promotion remain closed.
