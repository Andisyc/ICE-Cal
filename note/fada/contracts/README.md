# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v005](active/method/FADA-METHOD-v005.md) | method | active | Planner-IDM factorization with exact cold-start and scenario-preserving replay |
| [FADA-TRAIN-v005](active/training/FADA-TRAIN-v005.md) | training | active | Persistent-async UniLab training with scenario-balanced Planner replay |
| [FADA-CONTEXT-METHOD-v004](active/method/FADA-CONTEXT-METHOD-v004.md) | method | active | Fixed-0.7 Support-Query inverse-dynamics Context calibration |
| [FADA-CONTEXT-TRAIN-v003](active/training/FADA-CONTEXT-TRAIN-v003.md) | training | active | First-action Query supervision with only Context trainable |

The differentiable-dynamics route remains stopped history. The active Context route is the
human-confirmed Design Inspector 10 Support-Query method.

Superseded method/training contracts are retained under `history/` and excluded from default recall.
