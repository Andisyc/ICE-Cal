---
contract_id: FADA-METHOD-v010
status: active
effective_date: 2026-08-25
supersedes: FADA-METHOD-v009
scope: paper-ordered two-stage Planner-IDM training with permanent IDM freeze
---

# FADA Planner-IDM Method Contract v010

Authority: FADA Appendix B.2, the repository Planner-IDM design discussion, and the user's explicit
2026-08-25 decision to replace interleaved optimization with true two-stage training.

The deployable interface remains exactly `66/29/3`, `H=30`, `K=6`. Planner receives observation
history plus command and predicts a future-state chunk. That chunk enters the IDM Decoder path
directly; the IDM Encoder receives observation/action history, and only the first predicted action
executes.

## Phase authority

Training has exactly two ordered phases:

1. `idm_pretrain`: final/intermediate Oracle trajectories train only IDM. Planner parameters and
   optimizer are inactive. Main rollout is Oracle-owned for every iteration.
2. `planner`: a completed IDM-pretrain checkpoint is loaded by IDM weights only and permanently
   frozen. Planner action loss differentiates through the frozen IDM computation, but only Planner
   parameters update. DAgger uses Oracle rollout at iteration zero and current Planner-IDM rollout
   thereafter. Intermediate Oracle collection is disabled because those rows only train IDM.

No invocation may update both modules. Freezing IDM only during one Planner backward pass is not a
valid implementation of this contract.

## Source roles and replay

Schema-4 source rows retain the v009 `trajectory` and `oracle_shadow` IDM source-role contract.
Planner eligibility, scenario quotas `50/25/25`, cold-start quotas `50/50`, final walking/standing
Oracles, 20 intermediate walking Oracles, and 1:2 replay retention remain unchanged inside
`idm_pretrain`. Planner DAgger data never updates IDM and never enters the IDM-only 1:2 retention
route.

## Persistence

Training checkpoints use schema 4 and bind `training_phase`, `phase_completed`, exactly one
phase-owned optimizer, and the pretrained IDM identity used by Planner training. Planner phase
rejects missing, incomplete, wrong-phase, architecture-incompatible, identity-mismatched, or
output-path-aliased IDM checkpoints before runtime creation. v010 resume and warm start are disabled.
Inference-only playback continues to accept schema 1-3 checkpoints as historical evidence.

## Evidence boundary

Module, migration, and official offline route evidence are required before formal runtime audit.
Offline evidence does not claim convergence, walking stability, simulator quality, or authorize a
long training launch.
