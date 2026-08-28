# ICE-Cal documentation registry

This is the single entrypoint for repository-local documentation. The repository retains only
ICE-Cal architecture, contracts, plans, reviews, and evidence; general UniLab, G1 locomotion,
RoboJudo deployment, AMP, and unrelated distillation documentation belong to their source
repositories.

This registry owns research authority and dated evidence. Stable engineering explanations and
repeatable operator procedures live under [`docs/`](../docs/); they must link back here rather than
copying active Contract semantics.

## Current authority state

| Layer | Current artifact | Status |
|---|---|---|
| Concept Figure | `architecture/08_in_context_execution_calibration.html` | Synchronized with ICA-DP-08 |
| Design Inspector | `architecture/09_in_context_execution_calibration_design_inspector.html` | Human-confirmed v022 privileged source teacher + axis-bank calibration |
| Active FADA Contract lineage | `fada/contracts/README.md` | Source v022 plus active Context v009/v008 |
| Source engineering transition | `fada/plans/2026-08-29-fada-v022-grouped-dr-lineage.md` | Grouped-DR policy validated; sealed 20+1 lineage pending |
| Calibration engineering transition | `fada/plans/2026-08-23-configurable-axis-training-refactor.md` | Offline implemented and independently reviewed |
| Source governance | `fada/governance/2026-08-29-v022-privileged-teacher-design-confirmed.json` | v022 DESIGN-CONFIRMED; sealed lineage pending |
| Context governance | `governance.json` | CLOSED / OFFLINE EVIDENCE ONLY |

The Concept Figure and Inspector retain the data-driven perturbation basis, while source v022
defines how the privileged teacher and frozen Planner–Tracker are created and active Context
Contracts define one ordered active subset per calibration transaction. Fresh v008/v007
module evidence covers the local implementation; v007/v006 receipts remain historical. No offline
document authorizes training, simulation, deployment, or policy-quality claims.

## Recall order

1. Read this registry and `governance.json`.
2. Open the Concept Figure and Design Inspector under `architecture/`.
3. Read `fada/README.md`, then `fada/contracts/README.md` for current semantic authority.
4. Read the current engineering plan, Module Test Cards, checklist, and task canvas before any code
   work; use old evidence only for historical questions.

## Retained domains

- `architecture/`: ICE-Cal Concept Figure, Design Inspector, local Atlas runtime, and provenance.
- `fada/`: ICE-Cal/FADA contracts, plans, reviews, evidence, and task history.

No document in this registry authorizes training, live simulation, deployment, or Git publication.
