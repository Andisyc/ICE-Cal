# ICE-Cal documentation registry

This is the single entrypoint for repository-local documentation. The repository retains only
ICE-Cal architecture, contracts, plans, reviews, and evidence; general UniLab, G1 locomotion,
RoboJudo deployment, AMP, and unrelated distillation documentation belong to their source
repositories.

## Current authority state

| Layer | Current artifact | Status |
|---|---|---|
| Concept Figure | `architecture/08_in_context_execution_calibration.html` | Synchronized with ICA-DP-08 |
| Design Inspector | `architecture/09_in_context_execution_calibration_design_inspector.html` | Human-confirmed axis-bank + serial three-stage design |
| Active FADA Contract lineage | `fada/contracts/README.md` | Base v005 plus active Context v007/v006 |
| Engineering transition | `fada/plans/2026-08-19-calibratable-tracker-three-stage.md` | Implemented offline; module-correct; formal audit pending |
| Governance | `governance.json` | MODULE-CORRECT / FORMAL-AUDIT-PENDING |

The Concept Figure, Inspector, active Context Contracts, implementation, Module Test Cards, and
current offline receipts agree. The older query-conditioned module/final/formal receipts remain
historical. Current offline implementation evidence does not authorize training, simulation,
deployment, or policy-quality claims.

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
