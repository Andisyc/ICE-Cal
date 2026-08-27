# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v017](active/method/FADA-METHOD-v017.md) | method | active / design-confirmed | Perfect Oracle uses nominal dynamics only; failures belong to downstream frozen-backbone rollout collection |
| [FADA-TRAIN-v017](active/training/FADA-TRAIN-v017.md) | training | active / engineering-blocked | One nominal privileged-SAC lineage, then Planner–IDM, then isolated failure collection; no training authorization |
| [FADA-ADAPT-METHOD-v002](active/method/FADA-ADAPT-METHOD-v002.md) | method | active | IDM LoRA adaptation from non-leaking v2 target windows |
| [FADA-ADAPT-TRAIN-v002](active/training/FADA-ADAPT-TRAIN-v002.md) | training | active | v2 target split, LoRA-only optimization, adapted checkpoint, and admission |
| [FADA-CONTEXT-METHOD-v009](active/method/FADA-CONTEXT-METHOD-v009.md) | method | active | Data-driven task-relevant correction basis, frozen Tracker injection and coefficient readout |
| [FADA-CONTEXT-TRAIN-v008](active/training/FADA-CONTEXT-TRAIN-v008.md) | training | active | Serial basis discovery, operator freeze, coefficient training and scale evidence |

The base FADA v017 pair supersedes v016 because v016 leaked the downstream left-knee Gain calibration
variable into perfect-Oracle training. v017 keeps the single locomotion Reward, constant-zero phase
placeholders, and Planner–IDM tensor contract, but requires the complete Oracle lineage to be trained
under nominal dynamics only. Gain, delay, bias, and other failures begin only after the source
Oracle and Planner–Tracker are frozen. Current code still implements v016, so engineering, formal
audit, checkpoint reuse, server training, and policy-quality claims are blocked. The active Adapt
pair has the same evidence boundary.

The active Context pair is in semantic activation and bounded offline construction. v008/v007
analytic-axis implementation and review receipts became historical when the data-driven target and
component identity changed. Official-route formal
audit, simulator/training execution, and policy-quality evidence remain separate and have not run.

The superseded analytic-axis v008/v007 pair, fixed-three-axis v007/v006 pair, query-conditioned
v006/v005 pair, and receipts bound to them remain history. They cannot establish correctness for the
active data-driven Contracts.

Superseded method/training contracts, including v016, v015, v014, v013, v012, and v011, are retained under `history/` and excluded
from default recall.
