# DAgger Persistent Worker Metrics Connector HP-4a2a Evidence

Date: 2026-07-16

Scope: connect request-level persistent-worker timings to identity-free,
schema-validated HP-4 stage observations. No collector internals, parent
identity, JSON artifact, MuJoCo, training, A/B, or optimization is included.

## Boundary Decision

The worker cannot own complete A/B identity. Formal config hash and checkpoint
lineage belong to the parent workflow. Hashing checkpoint files again inside a
measured request would also contaminate the timing. HP-4a2a therefore emits
only stage observations; HP-4a2c must later attach immutable identity and
persist the run artifact.

## Core Parameter Trace

```text
request start
-> SharedWeightSync read: weight_sync
-> existing scenario collector: unchanged flat collect_seconds only
-> existing dataset save: artifact_write
-> request end: total_elapsed
-> DistillationStageObservation validation
-> DaggerCollectResult.metadata
```

The metadata envelope carries `performance_metrics_schema_version=1` and
exactly three observations. `weight_sync` has zero rows/steps;
`artifact_write` has dataset rows and zero env steps; `total_elapsed` has
dataset rows, existing dataset `env_steps`, and cleanup state `pending`.

The existing flat `metrics` mapping remains unchanged for compatibility.

## Test Evidence

Test-first red: both focused files failed to import
`DistillationStageObservation`.

Fake-clock facts for each of four semantic worker requests:

```text
weight_sync_seconds=0.1
artifact_write_seconds=0.25
total_elapsed_seconds=4.0
row_counts=(0, 4, 4)
role env_steps=2
transition env_steps=3
stage_order=(weight_sync, artifact_write, total_elapsed)
```

Fresh commands:

- focused schema + worker connector: `18 passed in 0.08s`;
- persistent runtime/workflow impact: `30 passed in 3.81s`;
- production factory selection: `2 passed, 195 deselected in 0.42s`;
- Ruff: `All checks passed!`.

The first impact command named a nonexistent test file and executed no tests;
it was corrected immediately. Only the corrected `30 passed` run is evidence.

## Decision

HP-4a2a passes its S1/S2 connector gate. Runtime integration is partial:
collector-level observations, parent identity enrichment, artifact persistence,
cleanup-final records, and live timing remain unconfirmed. HP-4a2b/2c and
HP-4b remain blocked pending separate user decisions.
