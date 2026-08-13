# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v005](active/method/FADA-METHOD-v005.md) | method | active | Planner-IDM factorization with exact cold-start and scenario-preserving replay |
| [FADA-TRAIN-v005](active/training/FADA-TRAIN-v005.md) | training | active | Persistent-async UniLab training with scenario-balanced Planner replay |
| [FADA-CONTEXT-METHOD-v005](active/method/FADA-CONTEXT-METHOD-v005.md) | method | active | Fixed-0.7 Support-Query calibration with complete-Query sliding windows |
| [FADA-CONTEXT-TRAIN-v004](active/training/FADA-CONTEXT-TRAIN-v004.md) | training | active | Pair-owned all-window first-action supervision with only Context trainable |

The differentiable-dynamics and single-anchor Query routes remain stopped history. The active Context
route is the human-confirmed Design Inspector 10 multi-sliding-window Support-Query method.

Superseded method/training contracts are retained under `history/` and excluded from default recall.
