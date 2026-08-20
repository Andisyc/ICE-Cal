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
- Completed engineering plan: `plans/2026-08-20-calibration-stage-isolation.md`
- Current checklist: `checklists/current.md`
- Confirmed Module Test Cards: `../testing/module_test_cards.md`
- Module receipt: `../testing/module_test_manifest.json`
- Final review: `reviews/2026-08-20-calibration-stage-isolation-final.json`

## Cursor

The v007/v006 library owners, preparation/evaluation/playback entrypoints, schemas, and
configuration remain implemented. Stage 1, Stage 2, and Stage 3 now have independent public
transactions and CLIs; later stages exact-byte load their predecessor artifact, while the serial
route crosses the same persisted boundaries. Current evidence is 17 owner rows, 75 semantic cases,
and a 151-test affected suite; Module Alignment and the maintainability final gate pass.

One serial-versus-independent Action-equivalence case is `CONNECTIVITY-ONLY`. The next technical
gate remains `formal-runtime-audit` through the official CLI/process, data, stage, persistence,
evaluation, and playback routes. It has not run and is not implied by module evidence.

## Preserved

Frozen Planner/Tracker; H=30; K=6; D=128; Decoder-only Action ownership; first-action-only
receding-horizon execution; historical evidence retained.

## Forbidden now

Simulator collection, long training, policy-quality evaluation, live playback/deployment, Git
writes, and reuse of v006/v005 receipts as v007/v006 evidence remain unauthorized.
