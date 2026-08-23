# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v005](active/method/FADA-METHOD-v005.md) | method | active | Planner-IDM factorization with exact cold-start and scenario-preserving replay |
| [FADA-TRAIN-v005](active/training/FADA-TRAIN-v005.md) | training | active | Persistent-async UniLab training with scenario-balanced Planner replay |
| [FADA-CONTEXT-METHOD-v008](active/method/FADA-CONTEXT-METHOD-v008.md) | method | active | Axis-bank latent calibration with a transaction-bound ordered active-axis subset |
| [FADA-CONTEXT-TRAIN-v007](active/training/FADA-CONTEXT-TRAIN-v007.md) | training | active | Configurable-width serial S1 direction, S2 coefficient, S3 scale-curve training |

The active Context pair is in bounded offline construction. v007/v006 implementation and review
receipts became historical when the fixed-three-axis training identity changed. Official-route formal
audit, simulator/training execution, and policy-quality evidence remain separate and have not run.

The superseded fixed-three-axis v007/v006 pair, query-conditioned v006/v005 pair, and receipts bound
to them remain history. They cannot establish correctness for the active configurable-axis Contracts.

Superseded method/training contracts are retained under `history/` and excluded from default recall.
