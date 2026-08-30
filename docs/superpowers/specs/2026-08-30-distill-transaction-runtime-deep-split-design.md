# Distillation Transaction and Playback Runtime Deep Split

## Goal

Finish the residual behavior-preserving split left by the previous owner extraction: each
collection/workflow entrypoint should compose explicit lifecycle phases instead of carrying the
whole transaction inline.

## Owners

- transition collection: one transaction owns rollout state, row accounting, and finalization;
- FADA collection: one transaction owns history, same-state Oracle ordering, accepted windows,
  rejection counters, and finalization;
- dataset merge: local pure phases own source loading/contract validation, concatenation, metadata,
  and final validation;
- DAgger: one typed iteration context owns collection, aggregation/update, metrics, and commit-last;
- interactive playback: the script keeps compatibility seams while explicit session preparation,
  viewer resources, loop, and cleanup phases own their local lifecycle.

## Invariants

- Public call signatures, tensor shapes/dtypes/devices, row/window order, reset handling, and
  scenario semantics stay unchanged.
- Oracle labels are queried on the pre-step state; DAgger manifest commit remains the last durable
  action.
- No config value, schema, checkpoint format, backend contract, trainer/reward math, or live route
  changes.
- Mutable state has one owner; extraction must not introduce a generic framework or duplicate
  environment/history/persistence ownership.

## Evidence boundary

Offline characterization, regression, and static checks may establish behavior preservation and
maintainability only. They do not establish simulator behavior, training convergence, checkpoint
quality, or policy quality.
