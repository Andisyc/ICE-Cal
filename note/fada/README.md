# FADA research registry

This directory records the ICE-Cal/FADA axis-bank calibratable-Tracker design, its Contracts,
engineering transition, and the superseded Support–Query implementation lineage.

## Current semantic boundary

Active `FADA-METHOD-v017` and `FADA-TRAIN-v017` own source-policy construction. They require one
ICE-Cal-trained privileged SAC Oracle on one G1WalkFlat/MuJoCo task and one locomotion Reward for
every command. The perfect Oracle is trained under strictly nominal dynamics: no left-knee Gain,
actuator attenuation, delay, bias, or other failure/domain randomization is present. Its 20
intermediate checkpoints and final checkpoint come from that one nominal lineage. Gain and every
other calibration fault belong only to downstream failed-rollout collection after the Oracle and
Planner–Tracker are frozen. The 98-D Actor, state66 + previous-action29 history, command3, action-free
K×66 future, and constant-zero gait-phase compatibility slots remain unchanged. v016 and earlier
source routes are historical. Current code still implements v016, so v017 engineering, formal
runtime, checkpoint reuse, simulation, training, and policy quality are blocked.

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
5. `plans/2026-08-27-fada-nominal-privileged-oracle-v017.md` for the current source-training
   correction; v016 plans, cards, receipts, and checkpoints are historical and cannot authorize v017
6. `plans/2026-08-23-configurable-axis-training-refactor.md`, `../testing/module_test_cards.md`, and
   `task_canvas.md` for the implemented calibration-side transition

Historical failed routes and old receipts are evidence, not policy-quality proof and not authority
to revive a superseded design.
