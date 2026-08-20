# FADA Task Canvas

Status: `IMPLEMENTED-OFFLINE / MODULE-CORRECT / FORMAL-AUDIT-PENDING`

## Objective

Implement the calibrated latent composition defined by Design Inspector 09 and active v007/v006:
`z_bar = z + sum_i sigma_i(c_i) Delta_z_i`.

## Current authority

- Concept Figure: `../architecture/08_in_context_execution_calibration.html`
- Design Inspector: `../architecture/09_in_context_execution_calibration_design_inspector.html`
- Method Contract: `contracts/active/method/FADA-CONTEXT-METHOD-v007.md`
- Training Contract: `contracts/active/training/FADA-CONTEXT-TRAIN-v006.md`
- Current plan: `plans/2026-08-19-calibratable-tracker-three-stage.md`
- Current checklist: `checklists/current.md`
- Confirmed Module Test Cards: `../testing/module_test_cards.md`
- Module receipt: `../testing/module_test_manifest.json`
- Final review: `reviews/2026-08-20-calibratable-tracker-final-gate.json`

## Cursor

The v007/v006 library owners, preparation/training/evaluation/playback entrypoints, schemas, and
configuration are implemented. Current offline evidence is 42 module tests plus a 211-test affected
regression suite; module admission and the maintainability final gate both pass.

The next technical gate is `formal-runtime-audit` through the official data, serial-stage,
persistence, evaluation, and playback routes. It has not run and is not implied by module evidence.

## Preserved

Frozen Planner/Tracker; H=30; K=6; D=128; Decoder-only Action ownership; first-action-only
receding-horizon execution; historical evidence retained.

## Forbidden now

Simulator collection, long training, policy-quality evaluation, live playback/deployment, Git
writes, and reuse of v006/v005 receipts as v007/v006 evidence remain unauthorized.
