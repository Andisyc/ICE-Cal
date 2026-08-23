# Configurable-Axis Final Review Object

Review date: 2026-08-23

## Identity

- Project: ICE-Cal
- Design: ICA-DP-01..10-calibratable-tracker-configurable-active-axes
- Contract: FADA-CONTEXT-METHOD-v008 + FADA-CONTEXT-TRAIN-v007
- Base: codex/in-context-execution-calibration@bdc4e420e39033558e5b1db9a03a018f4337bd5d
- Active code/test surface: content-sha256:4daeac6d96b3ae6454d72372e9971867692c276d3d7eee89b8b095eec81d9194
- Surface size: 35 files

The content identity is the SHA-256 of the ordered OpenSSL per-file SHA-256 output for the active
calibration configs, composition roots, library/package owners, and nine affected test files listed
by `note/testing/calibratable_tracker_module_test_evidence.json`.

## Requested Change

- Replace fixed `m=3` training state with one dataset-sealed ordered `CalibrationAxisSpec`.
- Support any non-empty registered subset while retaining full gain-delay-offset as the default.
- Split the 1214-line `calibration_training.py` into typed IO, lifecycle, Stage 1, Stage 2, Stage 3,
  and pipeline owners while preserving public imports.
- Isolate current trained schemas from retired state; admit only the exact raw-v1 gain donor through
  an explicit read-only Gateway.

## Reviewed Boundaries

- Dataset v2, Stage Artifact v3, Scale Evidence v2, and Final Artifact v2 each bind one canonical
  AxisSpec; mirrored architecture counts are validation-only.
- Only dataset preparation accepts repeated `--active-axis`; later stages consume the sealed type.
- Active raw v2 uses the YAML-owned catalog and rejects duplicate axis identity before publication.
- Stage IO owns hashing and atomic publication; lifecycle owns freeze/snapshot/rollback; stage modules
  own objectives and gates; pipeline owns serial composition.
- Evaluation compares dataset and artifact AxisSpec before runtime owner construction; m=1 combination
  evaluation rejects before downstream artifact loading.

## Resolved Review Findings

- Moved Stage 3 deployment publication from the stage module to the IO owner.
- Split active raw-v2 validation/writing from the exact read-only legacy-v1 Gateway.
- Added catalog injection at the collection Composition Root and removed production default-catalog use.
- Added reserved axis-identity rejection, unique temporary files, and failure cleanup at persistence owners.
- Added m=2 held-out role/order proof, same-width predecessor rejection, fresh reload, and zero nominal identity.
- Rebuilt Module Alignment and all current testing/Atlas entrypoints for v008/v007.

## Fresh Evidence

- Baseline before production edits: 157 tests passed.
- Final affected suite: 338 tests passed in 22.25 seconds.
- Independent reviewer rerun: 338 tests passed in 20.94 seconds.
- Ruff: All checks passed.
- mypy: Success, no issues in 23 source files.
- Module Alignment validator: ADMITTED-OFFLINE, 3 owner groups, 12 cases, 0 missing.
- Migration validator: offline contract PASS.
- `git diff --check`: clean.

## Verdict And Boundary

Independent final review: FINAL_GATE_PASS, with no remaining P0-P3 findings.

This object establishes offline maintainability, module, schema, persistence, and composition-root
evidence only. Formal runtime, simulator collection, real training, calibration efficacy, robustness,
policy quality, live playback, deployment, and real-robot behavior were not run and are not claimed.
