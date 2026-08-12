# FADA Task Canvas

Objective: train the healthy-simulation Tracker Encoder `E` and Decoder `D`, then freeze both. In
left-knee-strength-`0.9` simulation, use a faulty probe trajectory as Context input, emit latent
residual `delta_z`, execute a paired adapted rollout through the frozen Decoder, and optimize only
Context Encoder from reference-trajectory loss propagated through a learned differentiable fault
dynamics model. Deployment uses frozen `E/C/D` without weight updates or privileged fault data.

Current plan: `note/fada/plans/in_context_execution_calibration.md`

Active Context authority: `FADA-CONTEXT-METHOD-v003` and `FADA-CONTEXT-TRAIN-v002`. The former
direct-action-supervision route (`v002/v001`) and free-`delta_z` search-and-distill route are retained
as superseded/rejected design history, not current training authority.

Current cursor: method concept accepted on 2026-08-12; implementation and runtime validation have not
started. The accepted causal roles are:

- first faulty trajectory: causal, deployment-visible Context input;
- healthy same-command trajectory: reference target;
- second adapted trajectory: primary optimization result;
- learned fault dynamics ensemble: differentiable training-only gradient carrier;
- `E/D/F_phi`: frozen during Context updates; only `C` belongs to the optimizer;
- MuJoCo: fault data and transfer-validation owner, not an autograd path.

Completed prerequisite: formal paper-aligned v005 persistent-async Planner-IDM training completed
`8/8` on 2026-08-06, and its local checkpoint was hash-verified and strict-loaded. This does not
establish Context feasibility.

Evidence classes:

- `note-confirmed`: dual-rollout differentiable-fault-model Context design and parameter ownership;
- `contract-confirmed`: source Planner-IDM contracts end before target adaptation;
- `runtime-confirmed`: existing v005 checkpoint load only;
- `unconfirmed`: left-knee-`0.9` degradation for the selected `E/D`, dynamics-model accuracy,
  trajectory-gradient utility, Context identifiability, model-to-MuJoCo transfer, and deployment.

Next true boundary: specify the exact `E/D` checkpoint, probe/reference state schema, paired initial
condition lifecycle, short dynamics horizon, and numerical gates. No Context implementation or
training is authorized by this canvas alone.
