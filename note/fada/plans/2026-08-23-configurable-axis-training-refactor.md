# Configurable-Axis Calibration Training Refactor Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `code-construction` with
> `superpowers:test-driven-development`; obtain an independent `code-review-expert` READY gate before
> production edits and a final/migration gate after the coherent diff.

**Status:** offline implementation and independent review complete; formal runtime audit deferred. Supersedes
`2026-08-20-calibration-stage-isolation.md` for the calibration-training owner.

**Goal:** Replace the fixed-three-axis training assumption with one dataset-bound ordered active-axis
subset and split the 1214-line training module along the existing Stage 1/2/3 responsibility borders.

**Architecture:** `FaultAxisCatalog` owns registered cause definitions. `CalibrationAxisSpec` owns the
ordered, non-empty subset selected once during dataset sealing; later stages consume it from persisted
identity and cannot override it. The public `calibration_training` import becomes a package facade over
stage-specific modules, preserving caller imports while removing Divergent Change from the monolith.

**Tech stack:** Python 3.11, PyTorch, argparse, pytest, Ruff, mypy, `uv run`.

---

## Engineering boundary record

Requested behavior:

- select any non-empty ordered subset of registered axes at training-data sealing time;
- preserve caller order exactly, including non-catalog order such as `[offset, gain]`;
- default to `[gain, delay, offset]`;
- derive every S1/S2/S3 width, loop, tensor, gate, artifact and deployment owner from that selection;
- make the existing gain-only raw rollout resealable without simulator recollection;
- split calibration training by Stage responsibility while preserving its public imports.

Preserved behavior:

- registered gain/delay/offset definitions and analytic targets;
- frozen Planner/Tracker, H=30, K=6, D=128, one mutable owner per stage, existing losses and gates;
- serial persisted predecessor boundaries, exact-byte digest binding, rollback and atomic publication;
- six-Action decode and first-action-only execution;
- full three-axis selection remains the default behavior.

Forbidden behavior:

- no axis override in Stage 1/2/3 CLIs after dataset sealing;
- no silent reordering, duplicate/unknown/empty selection, zero-padding, or fixed-three-axis fallback;
- no loading old datasets, stage artifacts, scale evidence, or final artifacts into the new route;
- no simulator, training, deployment, network, commit, push, or policy-quality claim.

State/schema policy:

- current schemas: dataset v2, stage artifact v3, scale evidence v2, final artifact v2;
- old trained/persisted state rejects before mutation;
- the exact legacy v007/v006 gain raw schema is a read-only migration input and is resealed into a new
  dataset with `active_axes=[gain]`;
- artifact identity binds catalog version plus exact ordered active names.
- every new persisted envelope serializes exactly one canonical
  `axis_spec={catalog_version,names}` mapping; axis count is always derived from `names`;
- model configuration may mirror `axis_count` only as a tensor architecture check and must equal the
  reconstructed Axis Spec before model/optimizer construction.

Typed reconstruction boundary:

- `CalibrationAxisSpec.to_payload()` and `from_payload(payload, catalog=...)` are the only
  serializer/parser for the semantic identity;
- `load_calibration_dataset` returns `LoadedCalibrationDataset(batch, axis_spec, metadata)` rather
  than separately threading names/version/count primitives;
- `load_calibration_dataset(path, config, catalog)` and `load_calibration_artifact(path, catalog)`
  require the owner-YAML catalog at reconstruction; scripts load that catalog once at the Composition
  Root and pass the typed object inward;
- `CalibrationStageIdentity` contains `axis_spec` plus source/dataset/split digests;
- Stage Evidence, Stage Artifacts and the final artifact each carry one `axis_spec` payload and their
  loaders reconstruct/compare the Value Object before any mutable owner is created;
- no top-level `axis_names` plus metadata `axis_catalog_version` dual authority remains in new schemas.

Dependency direction:

```text
calibration.py (catalog, AxisSpec, models, deployment artifact)
  -> calibration_data.py (raw projection and dataset persistence)
  -> calibration_training/{types,io,lifecycle,stage1,stage2,stage3,pipeline}.py
  -> scripts (composition only)
```

Stage modules may depend on shared `types`, `io`, and `lifecycle`; `lifecycle` alone owns borrowed
module snapshot/freeze/restore checks. `pipeline` may depend on all stages. No stage module may import
`pipeline` or another stage's private implementation. Package `__init__.py` is the stable public
import boundary already consumed by scripts and tests.

## Task 1: RED axis-selection semantics

**Files:** `tests/algos/test_fada_calibration.py`,
`tests/algos/test_fada_calibration_training.py`,
`tests/scripts/test_fada_calibration_entrypoints.py`.

- [x] Add independent pseudo-samples for `[gain]`, non-catalog-order `[offset, gain]`, and full
  three-axis selections.
- [x] Require row filtering, axis-id remapping, `c_true` projection, and exact ordered identity.
- [x] For `[offset,gain]`, require projected columns, remapped IDs, Stage envelopes, Scale Evidence,
  final artifact, fresh reload and playback thresholds to retain that exact order; the catalog-sorted
  `[gain,offset]` predecessor is an identity mismatch and rejects before optimizer/publication.
- [x] Require empty, duplicate, unknown, and stage/dataset order mismatches to reject before optimizer
  construction or output publication.
- [x] Require gain-only Stage 1->2->3 shapes and artifacts to use `m=1`.
- [x] Require a complete `[offset,gain]` `m=2` owner round trip: projected batch, Direction Bank
  `[2,6,128]`, Encoder `[B,2]`, two compensation ratios, `[2,21,32]` Scale Evidence, final artifact,
  fresh reload and zero-coefficient nominal identity.
- [x] In the `m=2` projection, retain held-out rows containing only offset+gain and exclude rows with
  any delay coordinate; check the resulting role/axis identities independently.
- [x] Require the preparation parser to accept repeated `--active-axis`; later stage parsers must not.
- [x] Run the new nodes and observe RED because `CalibrationAxisSpec` and variable-width persistence do
  not yet exist.

## Task 2: Behavior-preserving training package split

**Files:** replace `src/unilab/algos/torch/fada_context/calibration_training.py` with package
`calibration_training/` containing `types.py`, `io.py`, `lifecycle.py`, `stage1.py`, `stage2.py`,
`stage3.py`, `pipeline.py`, and `__init__.py`.

- [x] Move immutable configurations/results/transaction identity into `types.py`.
- [x] Move hashing, atomic persistence, stage envelopes and scale evidence into `io.py`.
- [x] Move snapshot, freeze, rollback and source-projection validation into `lifecycle.py`.
- [x] Give each stage module its own objective, gate and transaction runner.
- [x] Keep serial composition only in `pipeline.py`.
- [x] Re-export the existing public names from package `__init__.py`; update no caller solely because of
  the split.
- [x] Run the fixed-three-axis characterization nodes after each move and keep them GREEN.

## Task 3: One AxisSpec across data and all stages

**Files:** `calibration.py`, `calibration_data.py`, the new training package, calibration scripts, and
their focused tests.

- [x] Add frozen `CalibrationAxisSpec(catalog_version, names)` with catalog-backed validation plus the
  single `to_payload`/`from_payload` persistence boundary.
- [x] Let `prepare_calibration_rollout_batch(..., axis_spec=...)` filter and project raw rows.
- [x] Seal one `axis_spec` object in dataset v2 and return `LoadedCalibrationDataset` with its typed
  reconstruction; metadata retains provenance only.
- [x] Replace every training-stage global axis-width assumption with
  `CalibrationStageIdentity.axis_spec`; no stage accepts names/version/count separately.
- [x] Make compensation ratios variable length and Stage 3 grids/evidence `[m,...]`.
- [x] Save/load deployment artifact v2 using its stored active-axis names; construct Direction Bank,
  Encoder and curves with `m` from the artifact.
- [x] Keep `FaultAxisCatalog.default()` only as a test/example constructor; production selection comes
  from owner YAML plus explicit `active_axes`.

## Task 4: Compatibility and owner-level CLI

**Files:** dataset preparation and S1/S2/S3/serial/scale-evidence scripts plus raw collection loader.

- [x] Add repeated `--active-axis` only to `prepare_fada_calibration_dataset.py`; omission selects the
  complete catalog in catalog order.
- [x] Make all later scripts consume `LoadedCalibrationDataset.axis_spec` and compare typed Axis Spec
  equality with predecessor artifacts/evidence. They must not reconstruct identity from metadata.
- [x] Pass the owner-YAML `FaultAxisCatalog` from every CLI Composition Root into dataset and final
  artifact reconstruction, including evaluation and playback. Later CLIs expose no active-axis override.
- [x] Keep metadata provenance-only and reject reserved axis-like metadata keys (`axis_spec`,
  `axis_names`, `axis_count`, `axis_catalog_version`) so they can neither define nor override identity.
- [x] Test tampered canonical Axis Spec rejection, reserved metadata rejection, and later CLI use of
  the typed loaded object rather than metadata primitives.
- [x] Make playback jump thresholds an axis-name mapping resolved in artifact order; remove the
  fixed three-value default from the playback script.
- [x] Keep held-out-combination evaluation explicitly unavailable for `m=1`; gain-only completion
  covers dataset and Stage 1->2->3/deployment artifact boundaries, not multi-axis evaluation.
- [x] Introduce a version-frozen read-only legacy raw Gateway with literal donor schema v1, v007/v006
  Contract IDs, `gain-delay-offset-v1`, exact axis order, gain-only labels, zero omitted coordinates,
  protocol/source/architecture/provenance digest checks. It validates before projection; dataset v2 is
  then published atomically.
- [x] Bump the active gain collector to raw schema v2/current Contract IDs. The v1 collector envelope
  is historical and cannot be written by the active writer; only the explicit Gateway reads it.
- [x] Test exact donor acceptance and altered schema/Contract/catalog/order, non-gain labels, nonzero
  omitted coordinates, source/protocol/backend digest and architecture rejection. All legacy trained
  schemas reject before optimizer or output creation.
- [x] Centralize repeated checkpoint/dataset/catalog/identity admission in one library helper only if it
  removes the current duplicated owner checks without moving business rules into scripts.

## Task 5: GREEN, migration matrix, and closeout

- [x] Run focused RED nodes, then the complete calibration test group.
- [x] Prove full-three-axis default equivalence, gain-only fresh reload, invalid selection rejection,
  atomic cleanup, old/new schema policy, frozen-owner preservation, and serial/independent equivalence.
- [x] Prove `[offset,gain]` end-to-end order preservation and reject a same-width sorted predecessor;
  tamper serialized version, names, derived model count and order independently.
- [x] Run Ruff formatting/check and mypy on changed production modules.
- [x] Run impacted-set Module Alignment and sensitivity cases.
- [x] Obtain independent `code-review-expert` migration/final-gate review; repair only same-scope findings.
- [x] Synchronize Contract registry, Inspector, checklist, task canvas, evidence, manifests and governance.

## Stop condition

Complete only when the active subset has one owner, the monolith no longer exists, all public imports
remain available, full-three-axis behavior stays the default, gain-only reaches all three offline stage
boundaries with `m=1`, old trained schemas fail closed, affected tests/static checks pass, and independent
review has no open P0/P1. Formal runtime, simulator execution, training efficacy, and policy quality remain
unclaimed.
