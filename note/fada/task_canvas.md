# FADA Task Canvas

Active objective: keep the healthy Planner-IDM checkpoint frozen, infer one condition-level `delta_z`
from a complete Support rollout, and train it from a disjoint fault-Query first-action label. Only
Context Encoder trains; deployment uses no fault labels or online optimizer updates.

Current plan: `note/fada/plans/in_context_execution_calibration.md`

Active Context authority: `FADA-CONTEXT-METHOD-v004` and `FADA-CONTEXT-TRAIN-v003`, implementing
Architecture 09 and Design Inspector 10. `FADA-CONTEXT-METHOD-v003` and
`FADA-CONTEXT-TRAIN-v002` remain stopped history.

Current cursor: implementation and bounded real-MuJoCo preflight are complete. The existing IDM final
hidden tokens are exposed as `z`; one Support-derived `[B,128]` condition vector is added immediately
before the frozen action head. The next gate is formal offline Context training, followed by a
separate fixed-Context fault-simulator trajectory evaluation.

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
- `unconfirmed`: full-dataset held-out action improvement and post-training deployment quality;
  cross-condition identifiability/generalization is intentionally outside the first fixed-`0.7`
  stage.

Next true boundary: run formal Context training only with explicit
`boundary.optimizer_steps_allowed=true`; then evaluate a fixed Support-derived `delta_z` in an
independent second fault rollout. No formal training is running now.
