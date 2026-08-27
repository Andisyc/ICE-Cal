# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v016](active/method/FADA-METHOD-v016.md) | method | active / module-correct | Single phase-neutral locomotion Reward, constant-zero compatibility slots, 95-D Planner history, and action-free 66-D future |
| [FADA-TRAIN-v016](active/training/FADA-TRAIN-v016.md) | training | active / formal-audit-pending | Nominal single-Reward gate, then gain-targeted privileged-SAC lineage; no training authorization |
| [FADA-ADAPT-METHOD-v002](active/method/FADA-ADAPT-METHOD-v002.md) | method | active | IDM LoRA adaptation from non-leaking v2 target windows |
| [FADA-ADAPT-TRAIN-v002](active/training/FADA-ADAPT-TRAIN-v002.md) | training | active | v2 target split, LoRA-only optimization, adapted checkpoint, and admission |
| [FADA-CONTEXT-METHOD-v009](active/method/FADA-CONTEXT-METHOD-v009.md) | method | active | Data-driven task-relevant correction basis, frozen Tracker injection and coefficient readout |
| [FADA-CONTEXT-TRAIN-v008](active/training/FADA-CONTEXT-TRAIN-v008.md) | training | active | Serial basis discovery, operator freeze, coefficient training and scale evidence |

The base FADA v016 pair supersedes v015 because live v015 training exposed early-termination Reward
hacking under its command-conditioned dual Reward. v016 uses one locomotion Reward for zero and
nonzero commands; Command changes the tracking target but does not select a Reward family. The two
legacy gait-phase slots remain constant zero and all phase Reward/constraints remain disabled.
The v016 source/config migration has current local module evidence. It must not be trained until a
separately authorized formal runtime audit and nominal policy-quality gate pass. The active Adapt
pair has the same evidence boundary.

The active Context pair is in semantic activation and bounded offline construction. v008/v007
analytic-axis implementation and review receipts became historical when the data-driven target and
component identity changed. Official-route formal
audit, simulator/training execution, and policy-quality evidence remain separate and have not run.

The superseded analytic-axis v008/v007 pair, fixed-three-axis v007/v006 pair, query-conditioned
v006/v005 pair, and receipts bound to them remain history. They cannot establish correctness for the
active data-driven Contracts.

Superseded method/training contracts, including v015, v014, v013, v012, and v011, are retained under `history/` and excluded
from default recall.
