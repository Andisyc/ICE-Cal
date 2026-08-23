---
contract_id: FADA-CONTEXT-METHOD-v008
status: active
effective_date: 2026-08-23
supersedes: FADA-CONTEXT-METHOD-v007
prerequisite: FADA-METHOD-v005
design_points: ICA-DP-01..10
scope: configurable-subset calibratable-Tracker direction bank, coefficient readout, and scale calibration
implementation_status: engineering-in-progress
---
# FADA Calibratable Tracker Method Contract

This Contract preserves the v007 calibratable-Tracker method and separates the registered fault-axis
catalog from the axes selected for one calibration-training transaction.

## Scientific object

The learned object remains a finite bank of execution-correction directions and a deployable readout
of their scalar strengths:

```text
z_bar = z + sum_i sigma_i(c_i) * Delta_z_i
action_chunk = FrozenTrackerDecoder(z_bar)
```

`Delta_z_i` is one normalized `[K,D]` direction field for active axis `i`; `c_i` is one scalar
coefficient; `sigma_i` is one frozen monotone calibration curve. The initial registered catalog is
`gain`, `delay`, and `offset`. One training transaction selects an ordered, non-empty subset
`active_axes` from that catalog, and `m = len(active_axes)` throughout its dataset, stages, artifacts,
evaluation, and deployment.

Selecting or removing an already registered axis is configuration. Registering a new physical cause
still requires an owned injection, normalized range, and analytic or independently validated target.

## Ownership

- Fault Axis Catalog owns registered causes, order-independent cause definitions, units, ranges,
  injections, analytic targets, and catalog version.
- Calibration Axis Spec owns the ordered `active_axes` subset for one complete Stage 1->2->3
  transaction. It rejects empty, duplicate, or unregistered axes.
- Planner owns Command-to-Future-Motion Intent and is always frozen.
- Tracker Encoder owns `z [B,K,D]`; Tracker Decoder remains the only Action generator. Both are
  always frozen.
- Direction Bank owns normalized `Delta_z_i [K,D]` fields learned only in Stage 1.
- Coefficient Encoder owns `c_hat [B,m]` from the current `H=30` State/Action history and is learned
  only in Stage 2.
- Scale Curve Bank owns frozen monotone `sigma_i` mappings fitted without neural-network training in
  Stage 3.
- Composition owner forms `z_bar`; no other owner may reinterpret or regenerate directions.

## Fault-axis and label contract

Fault enumeration is a training data-generation device, not a deployment input. The initial
registered catalog supplies these analytic Action targets:

```text
gain:   a_star = a_nominal / g
delay:  a_star = time-advanced a_nominal by the injected delay
offset: a_star = a_nominal - b
```

Dataset sealing projects raw catalog coordinates into the selected ordered subset. Single-axis rows
outside `active_axes` are excluded. A held-out combination is retained only when every nonzero cause
is selected and at least two selected axes remain.

## Deployment contract

At every control cycle:

1. Planner and Tracker Encoder produce nominal `z` from current histories and Intent.
2. Once the shared 30-frame State/Action window is full, Coefficient Encoder emits `[B,m]`; before
   that point all selected coefficients are zero.
3. Every in-range reading passes through its matching frozen `sigma_i`; range and jump events remain
   axis-identity bound.
4. The bounded selected-direction sum is added once before `decode_latent`; Decoder emits `K=6`
   Actions and receding-horizon execution consumes only index zero.

All modules and calibration artifacts are frozen during deployment. No Support/Query object, online
parameter update, direct Context Action, or hidden fault label is permitted.

## Required invariants

- `c=0` is exactly the nominal Tracker path for every valid active subset.
- Axis order is transaction identity: dataset, stage evidence, stage artifacts, final artifact, and
  deployment must bind the exact same ordered `active_axes`.
- One axis has one scalar coefficient but may have different directions for all six future tokens.
- Direction fields are normalized before publication so selected-axis scales are comparable.
- Addition is claimed only inside the covered first-order calibration region.
- Held-out multi-axis evidence is required only when `m >= 2`; a one-axis smoke transaction proves
  chain connectivity and the single-axis method case, not combination quality.
- Planner Intent and Tracker Action ownership do not change.

## Migration boundary

v007/v006 fixed-three-axis datasets, stage artifacts, scale evidence, final artifacts, and receipts
are historical and fail closed on the v008/v007 route. The already collected v007/v006 gain-only raw
rollout is admissible only through the explicit raw-to-v008 dataset resealing path because it retains
the exact registered-catalog provenance and contains no trained or deployable state. No simulator
recollection is required for that gain-only smoke case.
