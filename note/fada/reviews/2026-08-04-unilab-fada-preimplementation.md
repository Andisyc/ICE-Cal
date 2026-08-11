# UniLab FADA Migration Review - Preimplementation

Date: 2026-08-04
Mode: `migration_review`
Repository discipline: active (`AGENTS.md`, contract-first/config-first/owner-layer/backend-isolation)

Accepted boundary: default-off UniLab route that ends after source-domain Planner-IDM DAgger
training and paired checkpoint persistence. Target adaptation and oracle-shadow augmentation are
not in this unit.

## Route mapping verdict

No blocking donor-to-target mismatch was found. The existing composition root, public environment
factory, task-owned observation/command fields, and frozen SAC teacher are reusable. The existing
flat dataset, MLP/MoE trainer, and single-student checkpoint schemas are intentionally not reused
because doing so would create duplicated or ambiguous temporal and persistence ownership.

## Required isolation

- One flag: `training.fada.enabled`.
- OFF: dispatch remains byte-for-byte on the existing routing branches after the new false guard.
- ON: a dedicated FADA owner validates all architecture/collection/training parameters together.
- No backend-private snapshot access. Realized-trajectory IDM windows use only public `reset/step`
  outputs; oracle-shadow augmentation remains explicitly absent.
- `scripts/train_distill.py` remains a composition root and does not own window or gradient rules.

## Risk findings

No P0-P2 blocker before implementation.

P3: the formal script is already a large routing hotspot. Admission of a separate FADA workflow
function is acceptable only if algorithmic collection/training logic remains in distill package
owners and the script contains configuration translation plus dependency assembly only.

## Required proof

OFF dispatch characterization, ON dispatch/connectivity, command and episode provenance tests,
separate gradient-owner tests, paired checkpoint round-trip, stale flag search, focused lint/type,
and a postimplementation migration review.
