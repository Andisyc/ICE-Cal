# Current Semantic Objects
Identity: `ICA-DP-01..10-calibratable-tracker-serial-three-stage` /
`FADA-CONTEXT-METHOD-v007+FADA-CONTEXT-TRAIN-v006`.

| Object | Owner | Meaning |
|---|---|---|
| fault axis | Axis Catalog | identifiable execution-channel correction coordinate |
| `Delta_z_i [6,128]` | Direction Bank | frozen per-token latent correction direction |
| `c_hat_i` | Coefficient Encoder | raw current strength reading from 30-frame State/Action history |
| `sigma_i` | Scale Curve Bank | frozen monotone conversion from reading to direction strength |
| `direction_frozen` artifact | Stage 1 transaction | schema-v2 Direction-only owner envelope with fixed gate and transaction identity |
| `coefficient_frozen` artifact | Stage 2 transaction | schema-v2 Direction+Encoder envelope bound to the exact Stage 1 artifact bytes |
| Scale Evidence artifact | Scale Evidence persistence owner | canonical coefficient scan, 32-repeat readings/Action errors and transaction identity; Stage 3 hashes the same bytes it loads |
| deployment artifact | Stage 3 transaction | frozen directions, Encoder, curves, exact parent digest and exact Scale Evidence digest |
| calibrated latent | Composition policy | `z + sum_i sigma_i(c_i)Delta_z_i` |
| Action chunk | Frozen Tracker Decoder | six Actions; only index zero is executed |
Support/Query batches and unrestricted per-cycle `delta_z_t` are historical semantic objects.
