# FADA research registry

This directory records the ICE-Cal/FADA axis-bank calibratable-Tracker design, its Contracts,
engineering transition, and the superseded Support–Query implementation lineage.

## Current semantic boundary

Active `FADA-METHOD-v022` and `FADA-TRAIN-v022` own source-policy construction. The current teacher is
a live-privileged SAC Actor/Critic trained on one G1WalkFlat/MuJoCo task and one locomotion Reward.
Its typed privileged vector is normalized and fed consistently to Collector and Learner. An
iteration curriculum expands left-knee actuator strength together with Kp/Kd, friction, mass, COM,
and DoF-bias randomization; delay and pushes remain disabled. The 98-D task observation,
state66 + previous-action29 history, command3, action-free K×66 future, and constant-zero gait-phase
compatibility slots remain unchanged.

The `G1WalkFlat_live_priv_grouped_dr_v022` validation run qualitatively reached high Reward and
episode length. It is not yet the admissible source lineage: validation mode saves every 1000
iterations and does not produce the required `240…4800 + 5000` sealed checkpoints. Planner–IDM is
therefore persistence-blocked until the same successful curriculum is retrained through a sealed
20+1 profile. v017 and earlier source routes are historical.

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
5. `plans/2026-08-29-fada-v022-grouped-dr-lineage.md` for the current source-training persistence
   correction; v017 plans and earlier source receipts are historical
6. `plans/2026-08-23-configurable-axis-training-refactor.md`, `../testing/module_test_cards.md`, and
   `task_canvas.md` for the implemented calibration-side transition

Historical failed routes and old receipts are evidence, not policy-quality proof and not authority
to revive a superseded design.

The approved 15-degree narrow-slope reproduction has a separate operational
runbook at `docs/runbooks/fada-slope-traversal.md`. It keeps source training,
target-only collection, Q/V-only IDM LoRA adaptation, and same-snapshot
before/after evaluation as four distinct evidence boundaries.
