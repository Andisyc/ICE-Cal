# FT-0 Aggregate Workload Owner Integration

Date: 2026-07-17

## Scope

Integrate production replay-budget recomputation into the no-training FT-0
connector. No SSH, server materialization, supervisor, environment, collection,
learner, checkpoint, or training execution occurred.

## Owner Path

`parent aggregate .pt -> load_distillation_dataset() -> scenario_labels ->
required_balanced_replay_updates_for_labels() -> per-iteration required/effective
schedule -> spec comparison -> freeze failures/observed_workload`.

The replay formula remains owned by `offline.py`. The deploy connector only
loads the real aggregate, reads resolved Hydra scenario quotas/replay contract,
appends the configured scenario rows, and records the owner result.

## Fail-Closed Contract

- The connector does not trust the spec schedule as an observation.
- It recomputes aggregate rows, required/effective updates, and total updates
  from the real parent dataset and resolved config.
- Any schedule or total mismatch enters freeze failures and rejects preflight.
- Aggregate/compose failure becomes a workload observation error and rejects
  the spec comparison.
- The frozen training argv remains generated-only and is never invoked.

## Verification

- RED contracts exposed the absent workload observation and mismatch validator.
- A serialized `DistillationTensorDataset` fixture recomputes two rounds as
  rows `[15, 27]`, required/effective `[2, 4]`, and total `6` under a
  hand-checkable replay contract.
- Focused owner/connector/workflow/HP-7 regression: 23 passed.
- Targeted Ruff and mypy: PASS.

## Decision

The workload discriminator is now inside the one-line FT-0 materializer. Local
integration PASS. Server materialization is unexecuted and FT-1 remains closed.

