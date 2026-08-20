# FADA research registry

This directory records the ICE-Cal/FADA axis-bank calibratable-Tracker design, its Contracts,
engineering transition, and the superseded Support–Query implementation lineage.

## Current semantic boundary

Active `FADA-CONTEXT-METHOD-v007` and `FADA-CONTEXT-TRAIN-v006` define a frozen Planner and Tracker,
an axis direction bank, a 30-frame State/Action coefficient encoder, and serial S1/S2/S3 training.
Deployment composes `z + Σ σ_i(c_i)Δz_i`, decodes six Actions, and executes only Action zero.

The v007/v006 route is implemented offline with independent Stage 1/2/3 transactions, strict
predecessor artifacts, fresh module admission, and a maintainability review. The serial convenience
route crosses the same persisted boundaries. Formal official-route execution, simulator/training
evidence, and policy quality have not run. The former v006/v005 and pre-isolation receipts remain
historical and are invalid evidence for the current implementation identity.

## Recall order

1. `../README.md` and `../governance.json`
2. `../architecture/08_in_context_execution_calibration.html`
3. `../architecture/09_in_context_execution_calibration_design_inspector.html`
4. `contracts/README.md` for active semantic authority
5. `plans/2026-08-20-calibration-stage-isolation.md`, `../testing/module_test_cards.md`, and
   `task_canvas.md` for the engineering transition

Historical failed routes and old receipts are evidence, not policy-quality proof and not authority
to revive a superseded design.
