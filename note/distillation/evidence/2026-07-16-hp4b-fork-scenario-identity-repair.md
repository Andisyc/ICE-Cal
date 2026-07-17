# HP-4b Fork Scenario Identity Repair

Date: 2026-07-16

Status: `PASS` for the bounded schema-owner repair. Gate 0B and HP-4b remain
`BLOCKED` until a separately authorized refreeze and rerun.

## Authorized scope

Repair the first schema boundary exposed by E50 without changing the active
server, the r2 frozen cwd, existing datasets, scenario semantics, collection,
aggregation validation, or learner behavior.

## Root cause and owner

The active `DISTILL-TRAIN-v002` path already has an explicit data-owner upgrade:
`build_multitask_distillation_dataset()` uses each source mapping's `scenario`
to annotate legacy role rows in memory with complete scenario and transition
defaults. It does not rewrite the source artifact.

`fork_workflow_run()` reconstructed cumulative sources using only `path` and
`role`. It therefore discarded `WorkflowDatasetSource.scenario` and
`preserve_row_role_labels`. The next run could no longer invoke the existing
explicit annotation route, and mixed role/transition field presence reached the
fail-closed merge guard.

The owner-layer repair preserves both fields when a completed parent workflow
is forked. Data validation remains strict and no collector, script, legacy
artifact, or transition semantics changed.

## TDD and runtime-contract evidence

Red fixture before repair:

```text
test_multirole_dagger_scenario_manifest_and_quota_sources
KeyError: 'scenario'
```

Focused owner chain after repair:

```text
uv --cache-dir /private/tmp/uv-cache run pytest \
  tests/algos/test_distill_workflow.py::test_multirole_dagger_scenario_manifest_and_quota_sources \
  tests/algos/test_g1_distillation_contract.py::test_multitask_workflow_scenario_annotation_preserves_row_roles \
  tests/algos/test_g1_distillation_contract.py::test_multitask_distillation_dataset_merges_transition_fields -q
3 passed in 0.06s
```

The workflow fixture proves that the forked cumulative source list preserves
the exact scenario sequence and enables `preserve_row_role_labels` for every
source. It also hashes every parent source before and after the fork and proves
byte identity. The adjacent data-owner fixtures prove that these source fields
drive the existing in-memory legacy annotation and produce a transition-aware
merged dataset.

Affected suite:

```text
uv --cache-dir /private/tmp/uv-cache run pytest \
  tests/algos/test_distill_workflow.py \
  tests/algos/test_g1_distillation_contract.py \
  tests/scripts/test_train_scripts.py -q
288 passed, 8 skipped, 5 warnings in 6.36s
```

Static validation:

```text
uv --cache-dir /private/tmp/uv-cache run ruff check \
  src/unilab/algos/torch/distill/workflow.py \
  tests/algos/test_distill_workflow.py
All checks passed!
```

## Decision

The bounded schema-owner repair passes. It restores the intended chain:

```text
parent scenario artifacts
  -> WorkflowDatasetSource identity
  -> fork bootstrap source mappings
  -> explicit data-owner scenario annotation
  -> strict transition-aware cumulative merge
```

E49's source identity is now stale because mutable source and tests changed.
Therefore Gate 0B is `BLOCKED`, and E50's partial run is not resumable as a
formal A/B result. The next action is a separately authorized Gate 0B refreeze;
this step does not authorize refreeze, HP-4b execution, HP-4c, or a speedup
claim.
