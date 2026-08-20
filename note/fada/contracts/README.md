# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v005](active/method/FADA-METHOD-v005.md) | method | active | Planner-IDM factorization with exact cold-start and scenario-preserving replay |
| [FADA-TRAIN-v005](active/training/FADA-TRAIN-v005.md) | training | active | Persistent-async UniLab training with scenario-balanced Planner replay |
| [FADA-CONTEXT-METHOD-v007](active/method/FADA-CONTEXT-METHOD-v007.md) | method | active | Axis-bank latent calibration: frozen Tracker accepts `z + Σ σ_i(c_i)Δz_i` |
| [FADA-CONTEXT-TRAIN-v006](active/training/FADA-CONTEXT-TRAIN-v006.md) | training | active | Serial S1 direction, S2 coefficient, S3 scale-curve training; no joint training |

The active Context pair is implemented and admitted at the offline module layer. The current
maintainability final gate passes; official-route formal audit, simulator/training execution, and
policy-quality evidence remain separate and have not run.

The superseded query-conditioned Context versions v006/v005 and all receipts bound to them remain
history. They cannot establish correctness for the axis-bank calibratable-Tracker Contracts.

Superseded method/training contracts are retained under `history/` and excluded from default recall.
