# FADA Task Canvas

Objective: develop In-Context Execution Calibration after Planner-IDM construction. On the real
robot, collect a causal rollout, infer latent residual `delta_z`, add it to frozen Tracker latent
`z`, and decode `z + delta_z` with the frozen Decoder to produce precise actions without LoRA,
backpropagation, or any deployment weight update.

Historical concept figure (stale and non-authoritative after the latent-repair decision):
`note/architecture/architecture/08_trajectory_conditioned_execution_alignment.data.json`

Current proposal: `note/fada/plans/in_context_execution_calibration.md`

Source-training baseline: `note/architecture/concept/07_fada_planner_idm_distillation.data.json`

Active contracts: `FADA-METHOD-v005` and `FADA-TRAIN-v005` remain the completed Planner-IDM
prerequisite. `FADA-CONTEXT-METHOD-v001` governs the confirmed Context architecture:
`z_repaired = z + delta_z`, followed by the frozen Decoder. The v006 Phase-1 method/training
contracts govern the behavior-anchored full-action teacher retry. The v003/v004/v005 contracts are superseded history
and do not define the Context architecture.

Current step: v006 preserves the full 29D action teacher and fixed left-knee `0.9`, while repairing
the actor-update boundary that collapsed in v005. Training completed `1000/1000`; model 100 restored
400-step walking and improved forward progress/velocity, but formal quality failed because maximum
lateral displacement and yaw drift worsened. Formal v005 training completed `5000/5000`, but its teacher
terminated at step 50 in every row, advanced only `0.0282 m`, and failed at rate `1.0`. v004 completed `5000/5000` but
failed formal paired quality: it reduced maximum lateral/yaw error by `95.03%/65.96%` while producing
only `-0.0256 m` forward progress instead of `2.5806 m`. It remains rejected evidence. v003 also completed
`5000/5000` but failed formal quality. A subsequent frozen-nominal scan
showed that left-knee Kp/Kd multipliers `0.9`, `0.8`, and `0.7` do not degrade the walking policy
relative to `1.0`; all five seeds remained non-falling and generally became straighter. This
falsifies the current anomaly model before Context inference is tested. The deployed Context path
receives rollout history, never `g`, and remains blocked until a valid full-action teacher exists.

Completed prerequisite: formal paper-aligned v005 persistent-async Planner-IDM training completed
`8/8` on 2026-08-06 at `/ssd1/cyx/FADA_runs/20260806_planner_idm_v005`. The final checkpoint was
pulled to `/Users/sss9999/locomotion/FADA/planner_idm_v005.pt`, hash-verified, strict-loaded, and
produced a finite `(1,29)` inference action.

Active files: `FADA-CONTEXT-METHOD-v001`, the v006 Phase-1 method/training contracts, the current
Context plan/checklist, this task canvas, and the existing Planner-IDM contracts as an unchanged
prerequisite. The v003 contracts remain negative experimental history.

Verified evidence:

- `note-confirmed`: the human-confirmed method statement from 2026-08-07.
- `note-confirmed`: the 2026-08-11 decision that Context emits latent residual `delta_z`, the
  teacher emits a complete action, and only Context Encoder is trained during distillation.
- `contract-confirmed`: `FADA-METHOD-v005` and `FADA-TRAIN-v005` end before target adaptation.
- `runtime-confirmed`: v005 checkpoint pull, strict load, and finite single-action inference recorded
  by the existing FADA evidence ledger.

Unresolved risks: whether rollout history identifies the required latent repair, the exact Context
window, latent scaling/bounds, full-action teacher architecture, distillation loss, Context
lifecycle, cross-command validity, and which trajectory object defines precise control.

Next true boundary: choose a method that optimizes straight-line direction without losing v006's
restored gait. Context Encoder work remains blocked until a teacher passes the paired quality
prerequisite.
