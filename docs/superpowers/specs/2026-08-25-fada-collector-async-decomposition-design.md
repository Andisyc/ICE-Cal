# FADA Collector And Async Runtime Decomposition Design

## Goal

Split the two current FADA hotspots into responsibility-owned modules while preserving every
public import, tensor field, rollout decision, environment transaction, resource lifecycle,
artifact payload, checkpoint identity, and v010 phase behavior.

## Scope

This unit changes only `fada_collector.py`, `fada_async_runtime.py`, their new owner modules,
facade exports, and directly affected tests. It does not refactor `train_distill.py`, FADA
models/losses, replay, checkpoints, or the duplicated legacy/persistent training workflows.

## Preserved Public Boundaries

- `fada_collector.py` continues to export `FADACollectionResult`, `FADACollectionSpec`, and
  `collect_fada_source_windows` with the same object identities as their new owners.
- Historical private helper imports used by `fada_source_diagnostics.py` remain available from
  `fada_collector.py` during this migration.
- `fada_async_runtime.py` continues to export `FADA_ASYNC_SCENARIO`,
  `PersistentFADACollectorWorker`, `allocate_fada_command_scenarios`, and
  `build_persistent_fada_runtime`.
- Direct worker construction and the existing monkeypatch seams for `SharedWeightSync`,
  `BackendAdapter`, registry setup, environment creation, and teacher loading remain effective.

## Collector Owners

- `fada_collection_contract.py`: collection result/spec and causal transition record.
- `fada_collection_io.py`: observation, command, policy-action, done/reset, and Oracle-shadow
  environment transaction helpers.
- `fada_collection_windows.py`: pure transition-to-`FADASourceBatch` window builders and batch
  concatenation.
- `fada_collection_transaction.py`: the single environment rollout transaction and scenario
  coordination.
- `fada_collector.py`: compatibility facade only.

Dependency direction is contract -> IO/windows -> transaction -> facade. IO and windows may depend
on the FADA schema and generic projection/window helpers; none may depend on async runtime,
workflow, replay, trainer, checkpoint, scripts, or visualization.

## Async Owners

- `fada_async_config.py`: runtime device, stable-largest-remainder scenario allocation, teacher
  specification, curriculum defaults/validation, standing owner composition, and allocations.
- `fada_async_collection.py`: cold-start aggregation, per-request scenario/source collection,
  summary construction, source concatenation, and artifact commit transaction.
- `fada_async_runtime.py`: resident worker resource owner, cleanup, runtime factory, and historical
  facade exports.

The resident worker remains in `fada_async_runtime.py` so existing dependency-replacement seams
continue to target the owner that resolves them. Its `collect()` method delegates one request to
`fada_async_collection.collect_fada_iteration`, passing explicit resident resources.

## Failure And State Rules

- Oracle shadow always restores the exact visited-state snapshot on success and exception.
- Done/reset handling, pending forces, counters, episode identity, and command transitions do not
  move across a weaker boundary.
- Worker acquisition remains paired with fail-closed cleanup.
- Artifact save occurs only after all requested batches validate and carries unchanged metadata.
- No compatibility facade duplicates phase, scenario, tensor, resource, or persistence decisions.

## Verification

- RED structure tests require the new owner modules, facade symbol identity, forbidden reverse
  imports, and bounded facade sizes.
- Existing collector causal, scenario, cold-start, terminal, and Oracle-shadow tests remain green.
- Existing async startup/cleanup, weight sync, source allocation, artifact, and phase tests remain
  green.
- All FADA algorithm and Stage-C/D tests, Ruff, changed-module Pyright, import smoke, and
  `git diff --check` pass.

## Stop Boundary

This is a behavior-preserving offline refactor. It does not run MuJoCo, start training, access a
server, change schemas, retire legacy behavior, publish Git state, or claim policy quality.
