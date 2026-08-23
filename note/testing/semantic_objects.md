# Current Semantic Objects
Identity: `ICA-DP-01..10-calibratable-tracker-configurable-active-axes` /
`FADA-CONTEXT-METHOD-v008+FADA-CONTEXT-TRAIN-v007`.

| Object | Owner | Meaning |
|---|---|---|
| fault axis | Axis Catalog | identifiable execution-channel correction coordinate |
| ordered active axes | `CalibrationAxisSpec` | dataset-sealed catalog version plus exact non-empty axis order; `m=len(names)` |
| `Delta_z_i [6,128]` | Direction Bank | frozen per-token latent correction direction |
| `c_hat_i` | Coefficient Encoder | raw current strength reading from 30-frame State/Action history |
| `sigma_i` | Scale Curve Bank | frozen monotone conversion from reading to direction strength |
| `direction_frozen` artifact | Stage 1 transaction | schema-v3 Direction-only owner envelope with AxisSpec, fixed gate, and transaction identity |
| `coefficient_frozen` artifact | Stage 2 transaction | schema-v3 Direction+Encoder envelope bound to the exact Stage 1 artifact bytes and AxisSpec |
| Scale Evidence artifact | Scale Evidence persistence owner | canonical coefficient scan, 32-repeat readings/Action errors and transaction identity; Stage 3 hashes the same bytes it loads |
| deployment artifact | Stage 3 transaction | schema-v2 frozen directions, Encoder, curves, AxisSpec, parent digest and Scale Evidence digest |
| calibrated latent | Composition policy | `z + sum_i sigma_i(c_i)Delta_z_i` |
| Action chunk | Frozen Tracker Decoder | six Actions; only index zero is executed |
The exact raw-v1 gain envelope is a read-only Gateway donor. Fixed-width trained v007/v006 state,
Support/Query batches, and unrestricted per-cycle `delta_z_t` are historical semantic objects.
