---
contract_id: FADA-CONTEXT-METHOD-v007
status: historical
effective_date: 2026-08-19
supersedes: FADA-CONTEXT-METHOD-v006
prerequisite: FADA-METHOD-v005
design_points: ICA-DP-01..10
scope: serial calibratable-Tracker direction bank, coefficient readout, and scale calibration
implementation_status: implemented-module-correct-formal-not-run
---
# FADA Calibratable Tracker Method Contract

This active Contract projects the human-confirmed Design Inspector into semantic authority. It replaces
the complete-Support/query-conditioned free-residual route with a small, auditable calibration
coordinate system.

## Scientific object

The learned object is not an unrestricted environment embedding or a free action residual. It is a
finite bank of execution-correction directions and a deployable readout of their scalar strengths:

```text
z_bar = z + sum_i sigma_i(c_i) * Delta_z_i
action_chunk = FrozenTrackerDecoder(z_bar)
```

`Delta_z_i` is one normalized `[K,D]` direction field for fault axis `i`; `c_i` is one scalar
coefficient; `sigma_i` is one frozen monotone calibration curve. The initial axis library is
`gain`, `delay`, and `offset`.

## Ownership

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

Fault enumeration is a training data-generation device, not a deployment input. For the first
library, the simulator supplies exact fault strength and an analytic Action target:

```text
gain:   a_star = a_nominal / g
delay:  a_star = time-advanced a_nominal by the injected delay
offset: a_star = a_nominal - b
```

Each axis spans a declared normalized `c_true` range. Body dynamics, contact/terrain, observation
faults, and task/object parameters remain outside the first library unless a later Contract admits
an identifiable analytic or independently validated target.

## Deployment contract

At every control cycle:

1. Planner and Tracker Encoder produce the nominal `z` from the current histories and Intent.
2. Once the shared 30-frame State/Action window is full, Coefficient Encoder emits `c_hat`; before
   that point all coefficients are zero.
3. Every in-range reading passes through its frozen `sigma_i`; an out-of-range reading is visible,
   saturates only in the scale mapping, and raises an explicit range event.
4. A sudden adjacent reading jump raises a fast-failure event and freezes the affected axis rather
   than silently applying a discontinuous correction.
5. The bounded direction sum is added once before `decode_latent`; Decoder emits `K=6` Actions and
   receding-horizon execution consumes only index zero.

All modules and calibration artifacts are frozen during deployment. No Support/Query object, online
parameter update, direct Context Action, or hidden fault label is permitted.

## Required invariants

- `c=0` is exactly the nominal Tracker path.
- One axis has one scalar coefficient but may have different directions for all six future tokens.
- Direction fields are normalized before publication so axis scales are comparable.
- Addition is claimed only inside the covered first-order calibration region.
- Held-out multi-axis combinations are required evidence; single-axis success is insufficient.
- Planner Intent and Tracker Action ownership do not change.

## Migration boundary

The v006 complete-Support/query-conditioned `delta_z_t` implementation, schema-4 checkpoints,
Support/Query datasets, tests, formal receipts, and first-action proxy loss become historical. They
must not be silently loaded or cited as correctness evidence for v007.
