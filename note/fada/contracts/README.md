# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v005](active/method/FADA-METHOD-v005.md) | method | active | Planner-IDM factorization with exact cold-start and scenario-preserving replay |
| [FADA-TRAIN-v005](active/training/FADA-TRAIN-v005.md) | training | active | Persistent-async UniLab training with scenario-balanced Planner replay |
| [FADA-CONTEXT-METHOD-v003](active/method/FADA-CONTEXT-METHOD-v003.md) | method | active | Dual-rollout latent repair trained by trajectory loss through a differentiable fault model |
| [FADA-CONTEXT-TRAIN-v002](active/training/FADA-CONTEXT-TRAIN-v002.md) | training | active | Train only Context Encoder through frozen Tracker/Decoder/dynamics paths and validate in MuJoCo |

`FADA-CONTEXT-METHOD-v003` fixes the two-rollout roles, latent fusion, trajectory-level objective,
differentiable fault-model gradient path, parameter ownership, and MuJoCo aggregation requirement.
Exact runtime owners, schemas, architectures, horizons, weights, and thresholds remain open and
design-only.

Superseded method/training contracts are retained under `history/` and excluded from default recall.
