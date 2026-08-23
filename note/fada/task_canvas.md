# FADA Task Canvas

Status: `V008/V007 OFFLINE REVIEWED / FORMAL ROUTE DEFERRED`

## Objective

Replace fixed `m=3` calibration training with one dataset-bound ordered `active_axes` subset, while
splitting the Stage training monolith without changing the calibrated-Tracker mathematics.

## Current authority

- Concept Figure: `../architecture/08_in_context_execution_calibration.html`
- Design Inspector: `../architecture/09_in_context_execution_calibration_design_inspector.html`
- Method Contract: `contracts/active/method/FADA-CONTEXT-METHOD-v008.md`
- Training Contract: `contracts/active/training/FADA-CONTEXT-TRAIN-v007.md`
- Current plan: `plans/2026-08-23-configurable-axis-training-refactor.md`
- Plan review: `reviews/2026-08-23-configurable-axis-refactor-plan.json`
- Migration receipt: `reviews/2026-08-23-configurable-axis-migration-review.json`
- Final review: `reviews/2026-08-23-configurable-axis-final-gate.json`
- Migration manifest: `migrations/2026-08-23-configurable-axis-artifacts.json`
- Current checklist: `checklists/current.md`
- Confirmed Module Test Cards: `../testing/module_test_cards.md`

## Cursor

The configurable-axis implementation is complete in the local worktree. `CalibrationAxisSpec` is
sealed once in dataset v2; Stage 1/2/3, scale evidence v2, final artifact v2 and playback consume the
same ordered identity. The 1214-line training module is replaced by Stage-owned package modules.
Fresh impacted-set evidence is `338 passed`, Ruff is clean, and mypy reports no issues in the 23
source files. Module Alignment is `ADMITTED-OFFLINE`; migration and final review validators pass.
The next gated decision is a separately authorized formal runtime audit, not more offline refactoring.

## Preserved

Frozen Planner/Tracker; H=30; K=6; D=128; existing losses/gates; serial persisted stages; Decoder-only
Action ownership; first-action-only receding-horizon execution; full three-axis selection as default.

## Forbidden now

Simulator collection, long training, policy-quality evaluation, live playback/deployment, Git writes,
and promotion of offline module evidence into formal-route or efficacy claims remain unauthorized.
