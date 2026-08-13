---
contract_id: FADA-CONTEXT-TRAIN-v004
status: active
effective_date: 2026-08-13
updated_date: 2026-08-13
supersedes: FADA-CONTEXT-TRAIN-v003
method_contract: FADA-CONTEXT-METHOD-v005
scope: offline multi-sliding-window Context training under fixed left-knee strength 0.7
implementation_status: implemented-preflight-passed-training-not-started
---

# FADA Context Multi-Window Training Contract

Each accepted pair contains one complete Support rollout and one independent complete Query rollout.
The collector converts the Query into all valid `H=30`, `K=6` causal sliding windows. Context
Encoder processes Support once per sampled pair; the resulting fixed `delta_z` is reused across that
pair's windows. Training averages only valid first-action errors.

## Dataset and sampling

- Introduce a new dataset schema with explicit pair, window, anchor, and validity identities.
- Do not silently reinterpret or overwrite the previous single-anchor schema or artifacts.
- Split complete `(support_rollout_id, query_rollout_id)` groups before sampling windows.
- Sample pairs as the ownership unit. Within a sampled pair, use all valid windows or an explicitly
  configured unbiased window sample; record the choice in resolved config and checkpoint metadata.
- Weight pairs explicitly. A longer Query must not gain accidental weight merely because it has more
  padding or duplicated rows.

## Loss and gradient ownership

```text
delta_z[P,D] = ContextEncoder(Support[P,...])
delta_z -> broadcast across the owning pair's W windows and K latent tokens
prediction[P,W,K,A] = FrozenTracker(Query windows, delta_z)
loss = sum(mask * mse(prediction[:,:,0], executed_first_action)) / sum(mask)
```

The optimizer owns exactly Context Encoder parameters. Planner, Tracker Encoder, Tracker Decoder,
and normalizers remain in evaluation mode, excluded from the optimizer, and unchanged by every
checkpoint sentinel.

## Version and identity binding

The dataset and Context checkpoint bind:

- new dataset schema and complete FADA architecture;
- `H`, `K`, Query rollout length `L`, maximum/valid window counts, and window sampling policy;
- healthy source-checkpoint SHA-256 and Context architecture;
- task, command, fixed left-knee `0.7` provenance;
- Support/Query rollout identities and train/validation split identity;
- dataset, train-split, and validation-split SHA-256;
- zero-Context and trained all-window first-action MSE.

## Admission and stop gates

Long training is forbidden until focused tests and a bounded real-MuJoCo preflight prove the method
contract's alignment, masking, split, and freeze invariants. Stop on non-finite loss/gradient,
zero-identifiability signal, any frozen-parameter mutation, pair/window identity failure, or
train/validation rollout overlap.

Training acceptance and trajectory acceptance remain separate:

1. **Action-fit discriminator:** multi-window Context must reduce held-out all-window first-action MSE
   relative to both the single-anchor baseline and `delta_z=0`.
2. **Closed-loop discriminator:** fault plus fixed Context must be closer to the healthy checkpoint
   trajectory than fault plus zero Context, without worse falls or survival.

Passing only the first discriminator means improved action fitting, not fault repair.

## Current boundary

Pair-window schema v2, fixed `L=60` collection, masked first-action loss, rollout-group splitting,
checkpoint schema v3, and evaluation compatibility are implemented. Focused tests and a bounded
real-MuJoCo backward-only preflight passed with `optimizer_steps=0`. No formal multi-window Context
training has started; starting it still requires an explicit human command.
