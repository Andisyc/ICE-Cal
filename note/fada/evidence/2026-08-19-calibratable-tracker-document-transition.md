# Calibratable Tracker document transition

Date: 2026-08-19

## Authority change

The updated Design Inspector replaces the complete-Support/query-conditioned Context design with an
axis-bank calibratable-Tracker design:

- frozen Planner and Tracker;
- per-axis `[6,128]` direction fields and scalar coefficient readouts;
- a 30-frame State/Action Coefficient Encoder;
- serial S1 direction, S2 coefficient, and S3 monotone scale-curve training;
- `z + Σ σ_i(c_i)Δz_i`, six-step decoding, and first-action-only execution;
- no Support/Query runtime object and no joint training.

## Lifecycle actions

- Activated `FADA-CONTEXT-METHOD-v007` and `FADA-CONTEXT-TRAIN-v006`.
- Superseded v006/v005 Contracts were moved to history without deleting their evidence.
- Rebuilt the engineering plan, checklist, task canvas, semantic objects, impact rules, test
  inventory, control board, and Module Test Cards for the new design.
- Marked old module, maintainability, and formal receipts historical and non-current.
- Updated the Concept Figure and Design Inspector structural checks and Atlas manifest.

## Validation boundary

The Contract activation manifest passed `activate-contract`. The Atlas manifest and local standalone
page checks pass after synchronization. These are documentation and governance checks only.

No production code, test implementation, configuration, training, simulation, deployment, network
operation, or Git write was authorized or performed. Engineering remains transition-blocked until
the plan and Module Test Cards are reviewed and explicitly admitted.
