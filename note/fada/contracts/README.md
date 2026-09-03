# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v022](active/method/FADA-METHOD-v022.md) | method | active | Live normalized privileged Actor with iteration-based grouped DR curriculum, then Planner–IDM |
| [FADA-TRAIN-v022](active/training/FADA-TRAIN-v022.md) | training | active / persistence-blocked | Successful validation route exists; sealed 20+1 grouped-DR lineage is still missing |
| [FADA-ADAPT-METHOD-v003](active/method/FADA-ADAPT-METHOD-v003.md) | method | active | Q/V-only LoRA across all IDM attention from non-leaking target windows |
| [FADA-ADAPT-TRAIN-v003](active/training/FADA-ADAPT-TRAIN-v003.md) | training | active | Schema-5 source, v3 adapted checkpoint, legacy isolation, and 6000-step target budget |
| [FADA-CONTEXT-METHOD-v009](active/method/FADA-CONTEXT-METHOD-v009.md) | method | active | Data-driven task-relevant correction basis, frozen Tracker injection and coefficient readout |
| [FADA-CONTEXT-TRAIN-v008](active/training/FADA-CONTEXT-TRAIN-v008.md) | training | active | Serial basis discovery, operator freeze, coefficient training and scale evidence |

The base FADA v022 pair supersedes the v017 nominal-only teacher decision. The current source teacher
uses normalized live privileged information and an iteration-based grouped perturbation curriculum.
The observed `G1WalkFlat_live_priv_grouped_dr_v022` validation run reached high Reward and episode
length, but its validation profile saves every 1000 iterations and does not seal checkpoints. It
therefore cannot provide the required `240…4800 + 5000` IDM lineage. The next source-training change
is a sealed grouped-DR lineage profile, not another redesign of the successful curriculum.

The active Context pair is in semantic activation and bounded offline construction. v008/v007
analytic-axis implementation and review receipts became historical when the data-driven target and
component identity changed. Official-route formal
audit, simulator/training execution, and policy-quality evidence remain separate and have not run.

The superseded analytic-axis v008/v007 pair, fixed-three-axis v007/v006 pair, query-conditioned
v006/v005 pair, and receipts bound to them remain history. They cannot establish correctness for the
active data-driven Contracts.

Superseded method/training contracts, including FADA-ADAPT v002, v017, v016, v015, v014, v013, v012, and v011, are retained under `history/` and excluded
from default recall.

The 15-degree slope demonstration is an implementation of the active
FADA-ADAPT v003 contract. Its target artifact is target-only schema v3; legacy
actuator-gain schema v2 remains readable only through the explicit
`actuator_gain` compatibility boundary.
