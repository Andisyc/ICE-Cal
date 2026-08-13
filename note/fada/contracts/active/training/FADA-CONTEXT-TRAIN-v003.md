---
contract_id: FADA-CONTEXT-TRAIN-v003
status: active
effective_date: 2026-08-13
updated_date: 2026-08-13
supersedes: FADA-CONTEXT-TRAIN-v002
method_contract: FADA-CONTEXT-METHOD-v004
scope: offline support-query Context training under fixed left-knee strength 0.7
---

# FADA Context Support-Query Training Contract

The dataset owner runs the frozen healthy Planner-IDM twice in the same fixed-`0.7` fault environment.
Each accepted pair contains one complete Support trajectory and one independent causal Query window.
Rows that terminate, change command, contain non-finite values, or cannot supply a complete horizon
are rejected before dataset mutation.

The Context optimizer owns exactly Context Encoder parameters. The Planner and complete IDM are
strict-loaded from one healthy checkpoint, set to evaluation mode, and excluded from the optimizer.
Training computes Query latent tokens from Query realized future, adds one Support-derived `delta_z`,
decodes the action chunk, and applies MSE only to the first executed Query action.

The dataset and Context checkpoint bind:

- schema version and complete FADA architecture;
- healthy source-checkpoint SHA-256;
- exact task config and fixed actuator-strength identity;
- Support/Query pair identity and command equality;
- Context architecture and optimizer state;
- training/validation split seed and update cursor;
- zero-Context and trained action MSE.

Preflight permits collection, serialization round-trip, model construction, forward/backward without
optimizer step, and structured diagnostics. Long training starts only after those checks pass. A
completed action-supervision run remains insufficient for closed-loop acceptance; fixed-Context
fault-simulator evaluation is a separate required stage.
