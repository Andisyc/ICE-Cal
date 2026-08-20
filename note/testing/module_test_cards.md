# Calibratable Tracker v007/v006 Module Test Cards

Status: `CONFIRMED — user-authorized implementation execution, 2026-08-20`

These cards define intended public behavior. Implementation evidence is recorded separately and does
not by itself claim official-route connectivity, training quality, or deployment readiness.

## MTC-01 Axis Catalog and Analytic Labels

- Owner: fault-axis dataset/config owner.
- Input: registered axis, exact injected parameter, nominal six-step Action, provenance.
- Output: normalized `c_true`, analytic `a_star`, axis identity.
- Must prove: gain division, delay time advance, offset subtraction; finite/unit/range boundaries.
- Must reject: unknown axis, invalid gain zero, unavailable delay target, mixed identity or overlap.

## MTC-02 Direction Bank

- Owner: Direction Bank and Stage 1 trainer.
- Public object: normalized `[m,6,128]` fields plus scale provenance and admission status.
- Must prove: only current axis direction receives gradient; all frozen owners unchanged; `c=0`
  exact nominal identity; rejected axis is not published.
- Differential: shared `[128]` direction must fail the delay-token oracle that `[6,128]` satisfies.

## MTC-03 Coefficient Encoder

- Owner: rolling-history Coefficient Encoder.
- Input: `[B,30,O]` State history and `[B,30,A]` Action history from the Tracker buffer.
- Output: raw finite `[B,3]` coefficients; no clamp and no Action output.
- Must prove: both histories causally affect the relevant axis; row permutation covariance; exact
  zero-compatible initialization/cold-start owner; malformed history rejects before composition.

## MTC-04 Serial Stage Ownership

- Owner: three-stage training orchestrator in the library, not scripts.
- Must prove: Stage 1 mutates one direction only; Stage 2 mutates Encoder only; Stage 3 mutates only
  curve artifacts and constructs no optimizer.
- Must prove ordering gates, rollback on frozen mutation, exact checkpoint identity, and no joint
  direction/Encoder gradient path.

## MTC-05 Stage 2 Loss Semantics

- Owner: coefficient training loss.
- Oracle: `MSE(c_hat,c_true) + 0.1*MSE(action_chunk,a_star)`.
- Must prove coefficient term owns the minimizer in a controlled disagreement case; lambda is exactly
  `0.1`; Action safety path reaches frozen Decoder but not its parameters.
- Must reject non-finite/zero Encoder gradient and coefficient target mismatch.

## MTC-06 Scale Curve Bank

- Owner: monotone scale fitting and structured artifact loader.
- Must prove 21-point range, 32-repeat aggregation, PCHIP monotonicity, `R^2>=0.95`, endpoint mapping
  saturation with raw-reading preservation, and explicit out-of-range event.
- Must reject non-monotone data, insufficient grid, wrong axis/version, and silent extrapolation.

## MTC-07 Composition and Frozen Execution

- Owner: calibrated latent composition policy.
- Must prove `z + sum sigma(c)Delta_z` with axis/row/token identity, exact zero identity, bounded
  in-range correction, K=6 decode, and first Action only reaching the consumer.
- Must prove all Planner/Tracker/Encoder/Direction/Curve owners remain frozen per cycle.

## MTC-08 Readout Events and Cold Start

- Owner: deployment readout state machine.
- Must prove window-not-full gives exact zero, range event preserves raw reading while saturating
  mapping, jump event freezes only the affected axis, and reset clears history/event state.
- Invalid evidence must fail before Action consumption or use the declared fallback.

## MTC-09 Held-Out Combination

- Owner: evaluation transaction.
- Must hold out multi-axis combinations from single-axis construction and compare calibrated,
  nominal, wrong/shuffled coefficient, and full-finetune upper-bound routes.
- Must prove additive prediction and realized correction agree inside the declared first-order region;
  this evidence does not claim arbitrary unseen-cause coverage.

## MTC-10 Schema and Legacy Isolation

- Owner: v007/v006 dataset/checkpoint preparation.
- Must bind Contracts, catalog, H/K/D/m, normalization, Encoder, curves, stages, data/splits, source
  Tracker and provenance before construction.
- v006/v005 Support/Query datasets/checkpoints must reject before policy, optimizer, or mutable load.
