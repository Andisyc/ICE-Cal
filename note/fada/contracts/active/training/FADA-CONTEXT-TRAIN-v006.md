---
contract_id: FADA-CONTEXT-TRAIN-v006
status: active
effective_date: 2026-08-19
supersedes: FADA-CONTEXT-TRAIN-v005
method_contract: FADA-CONTEXT-METHOD-v007
scope: serial three-stage calibratable-Tracker construction
implementation_status: implemented-module-correct-formal-not-run
---
# FADA Calibratable Tracker Training Contract

## Shared labeled data

Stage 1 and Stage 2 reuse the same simulator rollouts. Every row binds axis identity, exact injected
strength, normalized `c_true`, nominal Action chunk, analytic `a_star`, current `H=30` State/Action
history, Planner Intent, and split/seed provenance. Train/validation identities and single-axis versus
held-out-combination roles remain explicit.

## Stage 1: Direction Bank

Train one axis at a time with Planner, Tracker, other directions, and Coefficient Encoder absent or
frozen. Substitute `c_true` directly:

```text
min_Delta || FrozenTracker(z + c_true * Delta_z_i) - a_star ||^2
```

Gradient may reach only the current `[K,D]` direction. Normalize it before freezing. Admit the axis
only when validation compensated error divided by uncompensated error is at most `0.1`.

## Stage 2: Coefficient Encoder

Freeze Planner, Tracker, and every direction. Train the two-layer `d_model=128` Transformer Encoder
from the shared 30-frame State/Action buffer. Mean-pool and linearly read out `m=3` scalars. The loss
is coefficient anchoring plus a low-authority Action consistency safety net:

```text
L = MSE(c_hat, c_true) + 0.1 * MSE(action_chunk(c_hat), a_star)
```

The coefficient term owns semantics. The Action term must not redefine coefficient values. Output is
not clamped. Admit only when normalized validation `|c_hat-c_true| <= 0.05` for the declared grid.

## Stage 3: Scale Curve Bank

No neural parameter is trained. For every axis, scan `c in [-1,1]` at 21 points with 32 rollouts per
point, fit a monotone PCHIP scale curve, and require `R^2 >= 0.95` plus monotonicity. Outside the
measured range, preserve the raw reading for audit, saturate the mapping at its endpoint, and raise an
explicit event. Real-robot recalibration, when separately authorized, uses known software-injected
faults and refits only `sigma`; it does not retrain the Encoder or Tracker.

## Gradient and persistence boundaries

- Stages are strictly serial; joint direction/coefficient training is forbidden.
- Each stage exposes exactly one mutable owner and verifies every frozen owner bitwise unchanged.
- Checkpoints bind Contract IDs, axis catalog/version, H/K/D, Direction Bank normalization, Encoder
  architecture, scale-curve grid, dataset/split digests, source Tracker digest, and stage status.
- A later stage cannot start until the preceding artifact passes its declared gate and is frozen.
- Non-finite data/loss/gradient, identity mismatch, frozen mutation, missing analytic target, or an
  unapproved axis stops before publication.

## Evidence boundary

Existing v006/v005 module, code-review, and formal receipts are historical only. New Module Test
Cards, a reviewed Engineering Plan, module correctness, formal-route evidence, and separate human
authority are required before training, simulation, or policy-quality claims.
