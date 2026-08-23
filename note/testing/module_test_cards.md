# Calibratable Tracker v008/v007 Module Test Cards

Status: `CONFIRMED — configurable-axis refactor authorized, 2026-08-23`

These cards define intended public behavior. Implementation evidence is recorded separately and does
not by itself claim official-route connectivity, training quality, or deployment readiness.

## MTC-01 Axis Catalog, Active Spec, and Analytic Labels

- Owner: Fault Axis Catalog owns registered causes; Calibration Axis Spec owns the ordered, non-empty
  subset selected for one complete training transaction.
- Input: catalog, ordered `active_axes`, exact injected parameter, nominal six-step Action, provenance.
- Output: projected `[B,m] c_true`, analytic `a_star`, remapped axis identity, exact ordered Axis Spec.
- Must prove: gain division, delay time advance, offset subtraction; finite/unit/range boundaries.
- Must prove: `[gain]`, non-catalog-order `[offset,gain]`, and default full-catalog selection; caller
  order is preserved through every persisted and runtime consumer; excluded row filtering and
  held-out combination retention only when every nonzero cause is selected.
- Must reject: empty/duplicate/unknown selection, invalid gain zero, unavailable delay target, mixed
  identity, silent reordering, or overlap.

## MTC-02 Direction Bank

- Owner: Direction Bank and Stage 1 trainer.
- Public object: normalized `[m,6,128]` fields plus scale provenance and admission status.
- Must prove: only current axis direction receives gradient; all frozen owners unchanged; `c=0`
  exact nominal identity; rejected axis is not published.
- Differential: shared `[128]` direction must fail the delay-token oracle that `[6,128]` satisfies.

## MTC-03 Coefficient Encoder

- Owner: rolling-history Coefficient Encoder.
- Input: `[B,30,O]` State history and `[B,30,A]` Action history from the Tracker buffer.
- Output: raw finite `[B,m]` coefficients in the persisted Axis Spec order; no clamp and no Action output.
- Must prove: both histories causally affect the relevant axis; row permutation covariance; exact
  zero-compatible initialization/cold-start owner; malformed history rejects before composition.

## MTC-04 Serial Stage Ownership

- Owner: three independent stage transactions in the library; scripts and the
  serial convenience route only compose them.
- Must prove: Stage 1 mutates one direction only; Stage 2 mutates Encoder only; Stage 3 mutates only
  curve artifacts and constructs no optimizer.
- Must prove: Stage 1 needs no future-stage input; Stage 2 can start only from a
  freshly loaded admitted Stage 1 artifact; Stage 3 can start only from a
  freshly loaded admitted Stage 2 artifact plus the exact typed Scale Evidence
  bytes it fingerprints.
- Must prove ordering gates, parent-digest identity, rollback on frozen mutation,
  serial-versus-independent fresh-reload equivalence, and no joint
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
- Must prove `[m,21]`, `[m,21,32]`, and candidate-axis shapes for `m=1`, `m=2`, and default `m=3`.

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
- For `m=1`, combination evaluation is explicitly not applicable and rejects before constructing the
  four comparison routes; gain-only smoke completion does not claim combination quality.

## MTC-10 Schema and Legacy Isolation

- Owner: v008/v007 dataset/artifact preparation.
- Must bind Contracts, catalog version, ordered `active_axes`, H/K/D/m, normalization, Encoder, curves, stages, data/splits, source
  Tracker and provenance before construction.
- The active stage envelope is a discriminated artifact: Stage 1 contains no
  random future Encoder, Stage 2 binds its Stage 1 parent digest, and the final
  artifact binds Stage 2 plus Scale Evidence digests. Stage publication uses a
  unique temporary sibling; failure exposes no new target, preserves an existing
  target byte-for-byte, and leaves no temporary residue.
- New schemas contain exactly one canonical `axis_spec={catalog_version,names}` identity reconstructed
  by the owner Value Object. Any mirrored model `axis_count` must equal `len(names)` before mutable
  construction. Tampered version, names, count, or order rejects independently.
- v007/v006 fixed-three-axis datasets, Stage Evidence, Stage Artifacts and final artifacts must reject
  before policy, optimizer, or mutable load. The exact legacy gain raw envelope alone may be resealed
  into a v008/v007 gain-only dataset; no trained state is migrated.
- The legacy raw Gateway freezes donor schema v1, v007/v006 IDs, catalog/order, gain-only labels,
  omitted-coordinate zeros, architecture and provenance digests; the active collector writes only
  current raw schema v2.
- Package refactoring must preserve the public `calibration_training` imports while tests patch the
  real stage/lifecycle/IO owner seam rather than facade module globals.
