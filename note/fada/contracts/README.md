# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v013](active/method/FADA-METHOD-v013.md) | method | active / locally implemented | Single privileged locomotion Oracle with command-conditioned stand/walk Reward, 95-D Planner history, and action-free 66-D future |
| [FADA-TRAIN-v013](active/training/FADA-TRAIN-v013.md) | training | active / module-evidence-only | Serial privileged-SAC lineage then fresh Planner–IDM campaign; no training authorization |
| [FADA-ADAPT-METHOD-v002](active/method/FADA-ADAPT-METHOD-v002.md) | method | active | IDM LoRA adaptation from non-leaking v2 target windows |
| [FADA-ADAPT-TRAIN-v002](active/training/FADA-ADAPT-TRAIN-v002.md) | training | active | v2 target split, LoRA-only optimization, adapted checkpoint, and admission |
| [FADA-CONTEXT-METHOD-v009](active/method/FADA-CONTEXT-METHOD-v009.md) | method | active | Data-driven task-relevant correction basis, frozen Tracker injection and coefficient readout |
| [FADA-CONTEXT-TRAIN-v008](active/training/FADA-CONTEXT-TRAIN-v008.md) | training | active | Serial basis discovery, operator freeze, coefficient training and scale evidence |

The base FADA v013 pair is locally implemented at the config/preflight boundary. v012 is historical
because its undifferentiated locomotion Reward could not distinguish a supported zero-command stand
from an invalid support geometry. v011 and earlier code and receipts are also historical. No prior
formal or module receipt can authorize v013 runtime or training. The active Adapt pair has the same
evidence boundary.

The active Context pair is in semantic activation and bounded offline construction. v008/v007
analytic-axis implementation and review receipts became historical when the data-driven target and
component identity changed. Official-route formal
audit, simulator/training execution, and policy-quality evidence remain separate and have not run.

The superseded analytic-axis v008/v007 pair, fixed-three-axis v007/v006 pair, query-conditioned
v006/v005 pair, and receipts bound to them remain history. They cannot establish correctness for the
active data-driven Contracts.

Superseded method/training contracts, including v012 and v011, are retained under `history/` and excluded
from default recall.
