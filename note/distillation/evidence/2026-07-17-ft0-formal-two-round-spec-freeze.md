# FT-0 Formal Two-Round Spec Freeze

Date: 2026-07-17

## Scope

Freeze and locally validate the reviewed formal workload/output spec only. No
materializer, Hydra compose, SSH, server artifact read/write, supervisor,
oracle preflight, environment, collection, learner, checkpoint, or training
execution occurred.

## Frozen Decision

- Parent lineage: original completed parent iteration 3.
- Added outer iterations: 2.
- Aggregate rows: iteration 1 `853504`; iteration 2 `855040`.
- Configured update floor: `512` per iteration.
- Required/effective schedule: `[12320, 12352]`.
- Total effective updates: `24672`.
- Seed/device: `0`, `cuda:0` with physical device selected later by Gate 0.
- Execution mode: explicit `persistent_async`; repository default unchanged.
- Output root:
  `/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_formal_dagger_2round_20260717_r1`.
- r6 sentinel checkpoint and supervisor are excluded.

## Owner Repair

The previous formal identity owner stored one scalar
`effective_updates_per_iteration`, which would have incorrectly represented two
rounds as `12320 * 2 = 24640`. A RED contract exposed the mismatch. The owner
now requires an immutable `effective_updates_by_iteration` schedule whose
length equals `dagger_iterations` and whose total is the exact sum. The
generated postflight oracle now compares every manifest iteration's `updates`
against the corresponding frozen schedule entry.

## Local Validation

`load_materialization_spec()` plus `build_formal_command_identity()` reports:

- `training_executed=false`;
- lineage source `original_parent_iteration_3`;
- `r6_sentinel_promoted=false`;
- `dagger_iterations=2`;
- schedule `[12320, 12352]`;
- total `24672`;
- the expected new absolute output root and owner-CLI argv.

## Boundary

This is a reviewed expected workload, not server-derived proof. The server
materializer must read the real parent aggregate and use the production replay
owner to recompute both per-iteration required/effective values. Any mismatch
blocks Gate 0. FT-1 remains closed.
