# Calibratable Tracker Serial Three-Stage Engineering Plan

Status: `DRAFT / TRANSITION-BLOCKED`

Design: `ICA-DP-01..10`

Contracts: `FADA-CONTEXT-METHOD-v007 + FADA-CONTEXT-TRAIN-v006`

This is the only current engineering plan for the calibratable-Tracker transition. It requires
human confirmation of the Module Test Cards and an independent `code-review-expert: READY` receipt
before production edits. It does not authorize training, simulation, policy evaluation, deployment,
or Git actions.

## 0. Preparation and migration boundary

Before the three learning stages, establish the new data and artifact boundary.

- Characterize the current v006/v005 loaders, policies, scripts, schemas, and tests.
- Define the v007 dataset and checkpoint identities.
- Reject v006/v005 Context artifacts before model construction or mutable state loading.
- Create the config-owned fault-axis catalog for `gain`, `delay`, and `offset`.
- Generate reusable labeled rollouts containing axis identity, `alpha_true`, `c_true`, nominal
  six-step Actions, analytic `a_star`, State/Action histories, Intent, seed, and split identity.
- Keep old implementation and receipts readable only as historical or explicit ablation paths.

Stop if the new route could silently reinterpret a Support/Query artifact.

## 1. Stage 1 — Direction Learning

**Purpose:** learn one latent correction direction for each declared fault axis.

The mutable owner is the Direction Bank:

```text
Delta_z_i in R^(6 x 128)
```

Execution:

- Use the analytic `c_true` for one axis at a time.
- Optimize only that axis direction.
- Keep Planner, Tracker, other directions, and Coefficient Encoder frozen.
- Normalize each admitted direction and persist its pre/post-normalization scale.
- Gate an axis on compensation ratio `<= 0.1`; exclude failed axes rather than tuning thresholds.

Stage 1 acceptance:

- `c=0` is bitwise equivalent to the nominal Tracker decode path.
- The direction shape is exactly `[6,128]`.
- Direction normalization preserves scale provenance.
- A failed axis cannot enter the Direction Bank as if it had passed.

After acceptance, freeze the complete Direction Bank.

## 2. Stage 2 — Coefficient Learning

**Purpose:** infer the current strength of each known axis from recent execution history.

The mutable owner is the Coefficient Encoder:

```text
c_hat = Encoder(StateHistory_30, ActionHistory_30)
```

Execution:

- Use the shared rolling 30-frame State and Action buffers.
- Implement a two-layer Transformer Encoder with `d_model=128`, mean pooling, and `Linear -> [B,m]`.
- Reuse Stage 1 labeled rollouts and keep the Direction Bank, Planner, and Tracker frozen.
- Optimize coefficient supervision plus the `lambda=0.1` Action safety term.
- Keep raw readings unclamped; range and jump handling remain explicit state outside the neural output.

Stage 2 acceptance:

- Normalized coefficient error is `<= 0.05`.
- Action loss cannot redefine the meaning of the coefficient target.
- The Encoder consumes only the declared 30-frame histories.
- Before 30 frames, the deployment readout is exact zero.

After acceptance, freeze the Coefficient Encoder.

## 3. Stage 3 — Scale Calibration

**Purpose:** convert each Encoder reading into the latent displacement scale required by the frozen
Direction Bank.

The mutable owner is the Scale Curve Bank, not a neural network:

```text
sigma_i: c_hat_i -> latent scale
```

Execution:

- Scan 21 points over `[-1,1]` and use 32 rollout repetitions per point.
- Fit one monotone PCHIP artifact per axis.
- Preserve raw Encoder readings and saturate only the mapped scale.
- Emit an explicit range event instead of silently extrapolating.
- Provide a calibration-only refit API that cannot mutate Tracker, Direction Bank, or Encoder.
- Bind field-refit artifacts to known software-injected fault provenance.

Stage 3 acceptance:

- The fitted curve is monotone with `R^2 >= 0.95`.
- Endpoint behavior is bounded and observable.
- Refitting changes only the Scale Curve artifact.
- No optimizer or gradient path reaches the frozen Tracker, Direction Bank, or Encoder.

After acceptance, freeze the complete Scale Curve Bank.

## 4. Composition and deployment

After all three stages are accepted and frozen, create one composition owner:

```text
z_bar = z + sum_i sigma_i(c_hat_i) * Delta_z_i
```

Deployment rules:

- Planner and Tracker remain frozen.
- Coefficients are recomputed every active control cycle.
- The Tracker decodes six Actions.
- Only Action zero is consumed before the next cycle.
- On jump or range failure, emit the typed event and apply the declared per-axis fallback.
- Prove the full chain remains frozen during deployment.

## 5. Combination, persistence, and admission

Validation and admission happen only after the implementation stages are complete.

- Hold out multi-axis combinations from single-axis construction and test additive recovery.
- Persist axis catalog, direction normalization, Encoder architecture, curves, stage gates, data
  digests, source Tracker digest, and Contract IDs in a new schema.
- Test exact resume at each stage; reject incomplete ordering and old schemas before mutation.
- Obtain a fresh `MODULE-CORRECT` receipt for v007/v006.
- Obtain final maintainability review.
- Run `formal-runtime-audit` on the real data, training, calibration, and deployment entrypoints.
- Request separate human authority before simulator training or policy-quality evaluation.

## Global stop conditions

- A stage updates more than its declared mutable owner.
- `c=0` changes nominal behavior.
- Direction normalization loses scale provenance.
- Coefficient semantics improve only by letting Action loss dominate.
- A scale curve is non-monotone or silently extrapolates.
- A held-out combination requires an unmodeled direct Action residual.
- A script becomes owner of axis semantics, loss, schema, or failure policy.
