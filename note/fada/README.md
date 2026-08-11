# FADA Workflow

Default recall order:

1. `contracts/README.md`
2. active method and training contracts
3. `../architecture/concept/04_fada_method_discussion.data.json`
4. `../architecture/concept/06_fada_design_detail_discussion.data.json`
5. current plan/checklist only when continuing active work

Current Context method authority: `FADA-CONTEXT-METHOD-v001`. It fixes latent repair as
`z_repaired = z + delta_z`, keeps Tracker Encoder and Decoder frozen during Context distillation,
and requires the privileged teacher to output a complete 29D action.

Current implementation status: v005 collapsed before its first saved checkpoint. Active v006 added
a frozen original-policy behavior anchor, restored full-horizon walking, and improved forward speed,
but failed formal paired quality because lateral displacement and yaw drift worsened. Its best
candidate, model 100, advanced `2.8192 m` with no falls but reached `0.1807 m` maximum lateral error
and `0.2673 rad` maximum yaw error. Context implementation remains blocked. Earlier residual routes
and v004/v005 full-action checkpoints remain failed evidence.

Phase-1 execution continues to use UniLab `DoubleBufferOffPolicyRunner` with synchronized collector
ticks and shared replay. v006 uses a behavior-anchored privileged full-action SAC learner; v002/v003
residual learners are rejected historical evidence. The completed Planner-IDM prerequisite uses `persistent_async`;
the two execution routes must not be conflated.
