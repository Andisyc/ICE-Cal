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
| Design Inspector | `architecture/09_in_context_execution_calibration_design_inspector.html` | Source v024 plus Context v009 design projection; no implementation claim |
| Project governance | `governance/README.md` | Separates Source Oracle, Adaptation and Context authority |
| Active FADA Contract lineage | `fada/contracts/README.md` | Source v024, Adapt v003 and Context v009/v008 |
| Source design transition | `fada/plans/2026-09-07-command-gated-phase-contact-oracle.md` | v024 implemented and offline-tested; formal runtime pending; not run |
| Source governance | `fada/governance/2026-09-07-v024-command-gated-phase-contact-design-confirmed.json` | v024 design activation receipt |
| Source module specification | `fada/testing/v024_module_test_cards.md` and `fada/testing/v024_reward_ordering_card.json` | Module oracles executed; six policy-ordering relations still require training evidence |
| Context governance | `governance.json` | Legacy v1 receipt for Context v009 activation only; not project-wide status |

The Concept Figure and Inspector project two connected but separately versioned objects: v024 owns
how the privileged source Oracle decides between standing and walking, while Context v009 owns how a
frozen Planner–Tracker is calibrated. Adapt v003 remains the target-domain LoRA route. Historical
v022/v023 source and v008/v007 Context evidence cannot authorize the current designs. No offline
document authorizes training, simulation, deployment, or policy-quality claims.

## Recall order

1. Read this registry and `governance/README.md`.
2. Open the Concept Figure and Design Inspector under `architecture/`.
3. Read `fada/README.md`, then `fada/contracts/README.md` for current semantic authority.
4. For Source Oracle work, read the v024 plan and Contracts. For Context work, read v009/v008.
5. Require a current engineering plan, Module Test Cards and impact scan before code work; use old
   evidence only for historical questions.

## Retained domains

- `architecture/`: ICE-Cal Concept Figure, Design Inspector, local Atlas runtime, and provenance.
- `fada/`: ICE-Cal/FADA contracts, plans, reviews, evidence, and task history.

No document in this registry authorizes training, live simulation, deployment, or Git publication.
