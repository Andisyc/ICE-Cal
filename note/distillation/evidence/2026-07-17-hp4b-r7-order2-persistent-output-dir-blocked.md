# HP-4b r7 Order 2 Persistent Output Directory Block

Date: 2026-07-17

Status: `BLOCKED` at order 2. Orders 3-8 did not start. No acceptance was
produced for order 2.

## Authorized Scope

Resume the frozen r7 A/B sequence at order 2, never rerun order 1, and invoke
the frozen acceptance oracle v2 after every successful run. Stop on the first
training or oracle failure.

Frozen identities:

- r7 execution identity SHA-256:
  `9b180b464433e0f29e59060c9245e9fbcd1879d988eeab802cee67be22f59718`.
- oracle v2 SHA-256:
  `9e62b678eb02d792c587b2a46ecc7fae1e000b9376d5bfbc229683170fedb631`.
- order 1 remained accepted and all seven contract-tracked order-1 artifact
  hashes matched before order 2 started.

## Runtime Facts

The first restricted launch stopped before rollout because POSIX shared memory
creation was denied by the execution sandbox. Its log and bootstrap-only
manifest were preserved outside the formal output root:

- log:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/execution_logs/order_02_persistent_async_sandbox_blocked.log`,
  SHA-256 `06c8abb260ba6905572c159f33bb7b33e3cdd47b2f6d4bbc159e731899b61d0a`;
- isolated partial run:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/execution_logs/order_02_sandbox_blocked_partial_run`,
  manifest SHA-256
  `d325566d5c50bb5690880868a132e4caf28cbf5200a9aeb5cd76ba46e7e6a39d`.

The exact frozen order-2 command was then retried with POSIX shared memory
available. SharedWeightSync and the spawned collector passed initialization.
The real persistent worker collected the first scenario, then failed at its
first artifact write:

```text
G1PersistentDaggerWorker.collect
-> save_distillation_dataset(request.output_path, dataset)
-> torch.save(...)
-> RuntimeError: Parent directory .../datasets/dagger_iteration_1 does not exist
```

Raw evidence:

- formal failure log:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/execution_logs/order_02_persistent_async.log`,
  SHA-256 `0b6fe888bd0a5b22ca4efece52729e7aa7717b08a0c23d80f79585e1885b7c67`;
- formal partial manifest:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_ab_20260717_gate0b_r7/persistent_r1/run_manifest.json`,
  SHA-256 `d325566d5c50bb5690880868a132e4caf28cbf5200a9aeb5cd76ba46e7e6a39d`.

The partial manifest remains at `BOOTSTRAP_COMPLETE` with zero completed DAgger
iterations. There are no scenario datasets, aggregate, new checkpoint,
distillation metrics, cleanup-final artifact, order-2 acceptance, or oracle
log. A post-failure process-table probe found no matching train, sequential
executor, or persistent collector process.

## Owner Diagnosis

This is a workflow output-materialization defect, not a dataset-schema,
teacher, checkpoint, SharedWeightSync, or oracle defect.

- `workflow.py` creates the logical `iteration_dir` path and dispatches both
  execution modes, but does not materialize that directory before dispatch.
- The legacy callbacks in `scripts/train_distill.py` create
  `output_path.parent`, which is why order 1 completed.
- `g1_persistent_worker.py` saves directly to the request output and correctly
  exposes the missing workflow-owned parent directory.
- Existing persistent workflow tests use `_write()`, whose helper creates the
  parent directory and therefore masks the formal-route defect.

The bounded repair should be owned by the workflow iteration materialization
boundary, with a regression proving that the output parent exists before a
persistent collector receives its first request. Do not add a script-only
mkdir or a persistent-worker-specific fallback.

## Stop Decision

The sequential executor stopped at the first failed training command as
required. Orders 3-8 were not started and HP-4c/HP-5 remain closed. Source was
not modified and order 2 was not retried after the owner defect was observed.
A separately authorized owner repair, focused regression, new source freeze,
and new formal output identity are required before restarting the A/B sequence.
