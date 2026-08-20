# Current Semantic Objects
Identity: `FADA-CONTEXT-METHOD-v007 + FADA-CONTEXT-TRAIN-v006`.
| Object | Owner | Meaning |
|---|---|---|
| fault axis | Axis Catalog | identifiable execution-channel correction coordinate |
| `Delta_z_i [6,128]` | Direction Bank | frozen per-token latent correction direction |
| `c_hat_i` | Coefficient Encoder | raw current strength reading from 30-frame State/Action history |
| `sigma_i` | Scale Curve Bank | frozen monotone conversion from reading to direction strength |
| calibrated latent | Composition policy | `z + sum_i sigma_i(c_i)Delta_z_i` |
| Action chunk | Frozen Tracker Decoder | six Actions; only index zero is executed |
Support/Query batches and unrestricted per-cycle `delta_z_t` are historical semantic objects.
