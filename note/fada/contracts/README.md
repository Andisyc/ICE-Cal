# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v005](active/method/FADA-METHOD-v005.md) | method | active | Planner-IDM factorization with exact cold-start and scenario-preserving replay |
| [FADA-TRAIN-v005](active/training/FADA-TRAIN-v005.md) | training | active | Persistent-async UniLab training with scenario-balanced Planner replay |
| [FADA-CONTEXT-METHOD-v001](active/method/FADA-CONTEXT-METHOD-v001.md) | method | active | Context Encoder latent residual `z_repaired = z + delta_z` before a frozen Decoder |
| [FADA-CONTEXT-PHASE1-METHOD-v006](active/method/FADA-CONTEXT-PHASE1-METHOD-v006.md) | method | active | Behavior-anchored fixed-left-knee-0.9 full-action teacher |
| [FADA-CONTEXT-PHASE1-TRAIN-v006](active/training/FADA-CONTEXT-PHASE1-TRAIN-v006.md) | training | active | Conservative SAC retry with early checkpoint discrimination |

`FADA-CONTEXT-METHOD-v001` fixes Context architecture and parameter ownership only. Its exact
training/runtime route, target-domain evaluation, LoRA, and later FADA stages remain outside active
training contracts.

Superseded method/training contracts are retained under `history/` and excluded from default recall.
