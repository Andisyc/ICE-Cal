---
contract_id: FADA-CONTEXT-TRAIN-v007
status: active
effective_date: 2026-08-23
supersedes: FADA-CONTEXT-TRAIN-v006
method_contract: FADA-CONTEXT-METHOD-v008
scope: configurable-axis serial three-stage calibratable-Tracker construction
implementation_status: engineering-in-progress
---
# FADA Calibratable Tracker Training Contract

## Training selection

Dataset preparation receives one ordered, non-empty `active_axes` list. Every name must be unique and
registered in the selected Fault Axis Catalog. The resulting immutable Calibration Axis Spec binds
the catalog version and exact ordered names. Its width `m = len(active_axes)` owns every later tensor,
model, gate, evidence, and artifact dimension. The default selection is the complete registered
`[gain, delay, offset]` catalog.

Later Stage 1/2/3 commands do not accept an independent axis override. They consume the Axis Spec
sealed by the dataset and reject predecessor evidence with a different catalog version, axis count,
or axis order before optimizer construction or publication.

## Shared labeled data

Stage 1 and Stage 2 reuse the same simulator rollouts. Dataset sealing retains only rows representable
by `active_axes`, remaps `axis_id` into selected order, and projects `c_true` to `[B,m]`. Every row
continues to bind exact injected strength, nominal Action chunk, analytic `a_star`, current `H=30`
State/Action history, Planner Intent, and split/seed provenance.

## Stage 1: Direction Bank

Train every selected axis independently with Planner, Tracker, other directions, and Coefficient
Encoder absent or frozen. Substitute `c_true` directly:

```text
min_Delta || FrozenTracker(z + c_true * Delta_z_i) - a_star ||^2
```

Gradient may reach only the current `[K,D]` direction. Normalize it before freezing. Admit each
selected axis only when validation compensated error divided by uncompensated error is at most `0.1`.
The published compensation-ratio vector has length `m`.

## Stage 2: Coefficient Encoder

Freeze Planner, Tracker, and every selected direction. Train the two-layer `d_model=128` Transformer
Encoder from the shared 30-frame State/Action buffer. Mean-pool and linearly read out `m` scalars:

```text
L = MSE(c_hat, c_true) + 0.1 * MSE(action_chunk(c_hat), a_star)
```

The coefficient term owns semantics. Output is not clamped. Admit only when normalized validation
`|c_hat-c_true| <= 0.05` for the selected axes.

## Stage 3: Scale Curve Bank

No neural parameter is trained. For every selected axis, scan `c in [-1,1]` at 21 points with 32
rollouts per point, fit one monotone PCHIP scale curve, and require `R^2 >= 0.95` plus monotonicity.
Scale Evidence shapes are `[m,21]`, `[m,21,32]`, and `[m,21,32,candidate]` as applicable.

## Gradient and persistence boundaries

- Stages remain strictly serial; joint direction/coefficient training is forbidden.
- Each stage exposes exactly one mutable owner and verifies every frozen owner bitwise unchanged.
- Dataset schema v2, Stage Artifact schema v3, Scale Evidence schema v2, and Calibration Artifact
  schema v2 bind Contract IDs, Calibration Axis Spec, H/K/D, model configuration, dataset/split/source
  digests, parent evidence digests, and stage status.
- A later stage cannot start until the preceding artifact passes its declared gate and exact Axis Spec.
- Non-finite data/loss/gradient, identity mismatch, frozen mutation, missing target, empty/duplicate/
  unknown axis selection, or axis-order mismatch stops before publication.
- Fixed-three-axis v007/v006 datasets and trained artifacts are rejected. Only the exact legacy
  gain-only raw rollout envelope may be resealed into a v008/v007 active-axis dataset.

## Evidence boundary

The prior v007/v006 module, code-review, and formal receipts are historical after this semantic and
schema change. New Module Test Cards, plan review, module correctness, migration review, and final
maintainability review are required. No training, simulation, deployment, or policy-quality claim is
authorized by this Contract.
