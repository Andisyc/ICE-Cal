---
contract_id: FADA-CONTEXT-METHOD-v009
status: active
effective_date: 2026-08-24
supersedes: FADA-CONTEXT-METHOD-v008
prerequisite: FADA-METHOD-v005
design_points: ICA-DP-01..07
scope: data-driven task-relevant execution-mismatch basis, frozen Tracker correction, coefficient readout, and scale calibration
implementation_status: engineering-proposal
---
# ICE-Cal Data-Driven Execution Calibration Method Contract

## Scientific object

The learned object is a finite bank of task-relevant latent correction operators discovered from
failure trajectories. Physical causes are data-generation and post-hoc interpretation variables;
they are not deployment coordinates. For a frozen Planner and Tracker, each rollout yields a
minimum-norm latent correction `delta_z_star`. Carrier regression followed by SVD extracts shared
operator components. Only components that pass the declared operator and held-out gates enter the
deployable bank.

```text
z_bar = z + sum_i sigma_i(c_i) * Delta_i(z)
action_chunk = FrozenTrackerDecoder(z_bar)
```

`Delta_i` is one normalized operator, `c_i` is one scalar readout coefficient, and `sigma_i` is
one frozen monotone scale mapping. The component index is an ordered transaction identity, not a
claim about a universal physical fault taxonomy.

## Ownership and evidence boundary

- Planner, Tracker Encoder and Tracker Decoder are frozen and remain the only Action generator.
- Basis Discovery creates `delta_z_star`, performs carrier regression and SVD, and records singular-
  value, residual and identification evidence; it does not publish an operator by itself.
- Stage 1 owns fitting one typed operator per admitted component and freezes it.
- Stage 2 owns the sole new network: a 2-layer `d_model=128` history Transformer producing `m` scalars.
- Stage 3 owns only monotone PCHIP scale artifacts; it creates no optimizer.
- Composition owns the single latent injection point before decoding.

## Causal data and target

Simulation collection records `(State/Action history, realized future, executed Action, theta)`.
The target uses the realized future to form `z_failed` and solves:

```text
delta_z_star = argmin_delta || Decoder(z_failed + delta) - executed_action ||^2
```

The solve starts at zero and retains the minimum-norm solution. The realized future is deliberately
not replaced by a predicted future: the resulting residual is the execution-mismatch signature.
Elementary gain/delay/offset formulas are unit-test or interpretation aids, not required supervision.

## Component admission and deployment

Carrier regression removes state-dependent scaling before SVD. A component is admitted only after
its typed operator reaches the shared-state compensation gate (ratio at least `0.9`), survives
held-out action-family checks, and has a declared scale normalization. Unidentified or mixed
components remain diagnostic-only and are not silently calibrated.

The Coefficient Encoder consumes the shared 30-frame State/Action history and outputs raw finite
`[B,m]` coefficients in the sealed component order. It does not output Actions, regenerate operators,
or receive deployment-time labels. `c=0` is the exact nominal path. Out-of-range readings are kept
raw, mapped with endpoint saturation, and raise an explicit alarm; no extrapolation is promised.

All additive composition claims are limited to the declared first-order coverage region. A held-out
multi-component combination is required whenever `m >= 2`; the first Gain-only transaction has
`m=1` and therefore does not claim combination quality.

## Real-machine calibration boundary

Real-machine data are used only to refit scale mappings and to monitor residuals; no network or
Tracker parameter is trained on the machine. Known software-injected test defects provide calibration
anchors. Unknown real defects are evaluation/monitoring cases, never circular calibration labels.

## Retirement boundary

`FADA-CONTEXT-METHOD-v008` and its training pair are historical analytic-axis contracts. Their
physical-axis labels, analytic `a_star` supervision and trained artifacts cannot authorize the v009
route. Legacy raw data may be inspected only through an explicitly versioned read-only adapter after
provenance and schema checks.
