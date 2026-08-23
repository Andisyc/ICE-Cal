# Current Checklist: Configurable-Axis Tracker v008/v007

Status: `OFFLINE-CLOSED / FORMAL-AUDIT-NOT-RUN`

## Authority

- [x] Design Inspector states `m=len(active_axes)` with full three-axis default.
- [x] `FADA-CONTEXT-METHOD-v008` and `FADA-CONTEXT-TRAIN-v007` active.
- [x] Fixed-three-axis v007/v006 Contracts and receipts classified historical.
- [x] Configurable-axis Module Test Cards confirmed by the user's execution authorization.
- [x] Engineering Plan received validated `code-review-expert: READY`.
- [x] State-schema migration plan validates.

## Engineering gates

- [x] One canonical `CalibrationAxisSpec` owns catalog version and ordered active names.
- [x] Dataset sealing filters/projects rows and preserves non-catalog caller order.
- [x] `calibration_training.py` is replaced by Stage-owned package modules.
- [x] Stage 1/2/3 and every persisted artifact derive `m` from the typed Axis Spec.
- [x] Exact legacy gain raw v1 reseals; old trained schemas reject before mutation.
- [x] Playback thresholds resolve by artifact axis names; `m=1` combination evaluation is N/A.
- [x] Full-three-axis default behavior remains characterized.

## Evidence gates

- [x] Existing fixed-three-axis baseline: 157 tests passed before production edits.
- [x] TDD RED cases observed for AxisSpec, dataset projection and package-owner boundaries.
- [x] Impacted-set Module Alignment current: 338 FADA calibration/context tests passed.
- [x] Ruff clean and mypy clean across 23 source files.
- [x] `code-review-expert: MIGRATION_REVIEW_PASS` and `FINAL_GATE_PASS` for the complete diff.
- [x] `formal-runtime-audit` remains separate and not run.
- [x] Training, simulation, deployment and policy-quality execution remain unauthorized.
