# FADA Task Canvas

Active objective: keep the healthy Planner-IDM checkpoint frozen, infer one condition-level `delta_z`
from a complete Support rollout, and train it from all causally valid first-action windows in a
disjoint complete fault Query. Only Context Encoder trains; deployment uses no fault labels or online
optimizer updates.

Current plan: `note/fada/plans/in_context_execution_calibration.md`

Active Context authority: `FADA-CONTEXT-METHOD-v005` and `FADA-CONTEXT-TRAIN-v004`, defining the
multi-sliding-window extension in Architecture 09 and Design Inspector 10. The implemented
single-anchor `v004/v003` route and differentiable-dynamics `v003/v002` route remain stopped history.

Current cursor: pair-window data, collection, dataset schema v2, fixed-`delta_z` broadcast, masked
first-action loss, checkpoint schema v3, and evaluation compatibility are implemented. Focused
tests and a bounded real-MuJoCo preflight passed. Formal training has not started and still requires
an explicit human command.

Completed prerequisite: formal paper-aligned v005 persistent-async Planner-IDM training completed
`8/8` on 2026-08-06, and its local checkpoint was hash-verified and strict-loaded. This does not
establish Context feasibility.

Evidence classes:

- `note-confirmed`: Support and Query share command and fixed fault but use independent rollouts;
  Context reads Support once and emits one fixed `delta_z`; Planner-IDM remains frozen;
- `contract-confirmed`: source Planner-IDM contracts end before target adaptation;
- `runtime-confirmed`: existing v005 checkpoint strict-load; fixed-`0.7` independent Support/Query
  collection; dataset round-trip; zero-Context first-action MSE `1.8847e-05`; Context gradient norm
  `1.7284e-04`; Planner/IDM frozen; one-step `/tmp` checkpoint smoke run;
- `runtime-confirmed`: the stopped route completed 10 rounds at left-knee strength `0.7`; every
  candidate worsened real-MuJoCo trajectory MSE and was rolled back;
- `runtime-confirmed`: the single-anchor Support-Query checkpoint completed training; fixed dataset
  Support and same-fault online Support both worsened all seven healthy-trajectory distance metrics;
- `note-confirmed`: complete Query becomes pair-owned sliding windows; one Support-derived fixed
  `delta_z` is reused over every valid Query window; first-action losses are averaged;
- `runtime-confirmed`: `L=60`, `H=30`, `K=6` produces 26 windows per Query at anchors `29..54`;
  39 focused/entrypoint tests passed; an 8-pair MuJoCo preflight accepted 208 windows with
  zero-Context MSE `2.3850443540140986e-05`, nonzero Context gradient, frozen Planner/IDM, and zero
  optimizer steps;
- `unconfirmed`: held-out all-window action improvement and post-training trajectory quality.
  Cross-condition generalization remains out of scope.

Next true boundary: after explicit authorization, start the formal multi-window Context run. The
first-stage Query length is fixed at `L=60`. No multi-window training is running now.
