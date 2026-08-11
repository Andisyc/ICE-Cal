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

Current implementation status: v005 fixed-left-knee-0.9 privileged full-action teacher retry
completed under `FADA-CONTEXT-PHASE1-METHOD-v005` and `FADA-CONTEXT-PHASE1-TRAIN-v005`, but failed
formal paired quality. It retained the
complete 29D action teacher and adds a default-off forward-progress failure termination to reject
the stationary shortcut. Threshold calibration, local/remote tests, a real MuJoCo discriminator,
and CUDA no-training preflight passed, but the trained policy reached only `0.0282 m` and terminated
at step 50 in every formal evaluation row. Context implementation remains blocked. The v002/v003
residual routes and v004/v005 full-action checkpoints are retained as failed evidence.

Phase-1 execution continues to use UniLab `DoubleBufferOffPolicyRunner` with synchronized collector
ticks and shared replay. v005 uses a privileged full-action SAC learner; v002/v003 residual learners
are rejected historical evidence. The completed Planner-IDM prerequisite uses `persistent_async`;
the two execution routes must not be conflated.
