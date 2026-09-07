# ICE-Cal governance index

This is the project-level governance entrypoint. It separates semantic objects that share the FADA
runtime but have different Contracts and evidence. A status in one row never authorizes another row.

| Track | Semantic owner | Governance cursor | Implementation/runtime boundary |
|---|---|---|---|
| Source Oracle | `FADA-METHOD-v024` + `FADA-TRAIN-v024` | design confirmed; engineering blocked | no v024 selector, code, formal route, checkpoint, training or quality evidence |
| Target adaptation | `FADA-ADAPT-METHOD-v003` + `FADA-ADAPT-TRAIN-v003` | active Contract | evidence is valid only for its recorded source/target/checkpoint identity |
| In-context calibration | `FADA-CONTEXT-METHOD-v009` + `FADA-CONTEXT-TRAIN-v008` | design confirmed; engineering proposal | v008/v007 implementation receipts are historical after semantic supersession |
| Repository simplification | `2026-09-07-repository-simplification.md` | offline execution complete, uncommitted | maintainability/config evidence only; no runtime or policy claim |

## Current receipts

- Source v024: `../fada/governance/2026-09-07-v024-command-gated-phase-contact-design-confirmed.json`
- Context v009: `../governance.json` is the preserved workflow-governance/v1 activation receipt.
  It is Context-specific and cannot serve as project-wide status or v2 engineering admission.
- Repository simplification: `../fada/governance/2026-09-07-repository-simplification-execution.json`.

## Authority order

1. Current explicit human instruction and the active Contract for the affected track.
2. Current Design Inspector projection and governance receipt.
3. Current engineering plan and producer-owned technical receipts.
4. Current implementation/configuration evidence.
5. Historical Contracts, plans, logs and memory.

The active Contract registry is `../fada/contracts/README.md`. Historical files remain evidence but
are excluded from default recall and cannot authorize implementation, training or deployment.
