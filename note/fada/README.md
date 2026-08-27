# FADA research registry

This directory records the ICE-Cal/FADA axis-bank calibratable-Tracker design, its Contracts,
engineering transition, and the superseded Support–Query implementation lineage.

## Current semantic boundary

Active `FADA-METHOD-v015` and `FADA-TRAIN-v015` own source-policy construction. They require one
ICE-Cal-trained privileged SAC Oracle on one G1WalkFlat/MuJoCo task. Zero command receives standing
support/stability Reward and nonzero command receives walking tracking Reward; the command is the
mode authority, while the two legacy gait-phase slots remain constant zero and all Gait/feet-phase
Reward and constraints remain disabled. A nominal standard-SAC profile validates this task before
privileged/Gain training but never joins the Oracle lineage. The final source
distribution is nominal plus left-knee actuator attenuation at index 3 (`g in [0.8,1.0]`, nominal
probability 0.3); unrelated physical DR is disabled. They bind one 20+1 checkpoint lineage and keep
Planner input as state66 + previous-action29 history with command3 separate and an action-free K×66
future. v014 and earlier source routes are historical.

Active `FADA-CONTEXT-METHOD-v008` and `FADA-CONTEXT-TRAIN-v007` define a frozen Planner and Tracker,
an axis direction bank, a 30-frame State/Action coefficient encoder, and serial S1/S2/S3 training.
The registered catalog initially contains gain/delay/offset; each dataset seals one ordered active
subset and all stage widths use `m=len(active_axes)`.
Deployment composes `z + Σ σ_i(c_i)Δz_i`, decodes six Actions, and executes only Action zero.

The prior fixed-three-axis v007/v006 route remains historical implementation evidence. The v008/v007
configurable-axis refactor is locally implemented with fresh offline module evidence and validated
migration/final maintainability receipts. Formal official-route execution, simulator/training
evidence, and policy quality have not run.

## Recall order

1. `../README.md` and `../governance.json`
2. `../architecture/08_in_context_execution_calibration.html`
3. `../architecture/09_in_context_execution_calibration_design_inspector.html`
4. `contracts/README.md` for active semantic authority
5. `plans/2026-08-27-fada-phase-neutral-dual-reward-v015.md` and
   `testing/v015_module_test_cards.md` for the current source-training transition
6. `plans/2026-08-23-configurable-axis-training-refactor.md`, `../testing/module_test_cards.md`, and
   `task_canvas.md` for the implemented calibration-side transition

Historical failed routes and old receipts are evidence, not policy-quality proof and not authority
to revive a superseded design.
