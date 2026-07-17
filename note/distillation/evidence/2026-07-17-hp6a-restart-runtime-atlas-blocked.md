# HP-6a Restart Blocked by Runtime Atlas Status Drift

Date: 2026-07-17

Evidence ID: E70

Status: BLOCKED at cross-file consistency after all affected tests passed.

## Scope Executed

- Repeated owner/default/lifecycle/lineage review with a structured static
  probe.
- Ran affected distillation algorithm, script/config, and IPC suites.
- Ran targeted Ruff on the cumulative affected Python surface.
- Reviewed current testing and Architecture surfaces after E69.

No training, benchmark, `make test-all`, contract activation, default-on,
commit, or PR action occurred.

## Valid Evidence

- Owner probe: all 10 facts true. Hydra remains `legacy`; script consumes the
  production factory interface without owning IPC primitives; distillation
  reuses `AsyncRunner` and `SharedWeightSync`; workflow owns execution-mode
  validation and iteration materialization; source status records OFF-default
  and `NO_STABLE_SPEEDUP`.
- Affected algorithm suite: 137 passed, 5 warnings.
- Script/config suite: 326 passed.
- IPC suite: 74 passed, 24 skipped.
- Total: 537 passed, 24 skipped.
- Targeted Ruff: all checks passed.

The first sandboxed algorithm run produced 3 `shm_open` permission failures and
134 passes. The exact command was rerun with shared-memory permission and
passed 137/137; only the latter is the functional verdict.

## Cross-File Blocker

The E69 stale assertion covered `absent`, `尚未`, and `未连接`, but not the
equivalent phrase `尚缺`. Fresh Runtime Atlas review found:

- `note/architecture/runtime/01_unilab_runtime_atlas.data.json:168` (U-RT-06)
  still says reset/resource live timing and A/B are missing.
- The same file at line 221 (U-RT-08) repeats that stale claim.

Both contradict E61/E65/E67. The atlas schema check can prove valid structure
and source mappings, but it does not detect this semantic evidence drift.

## Decision

E69 remains a valid local repair for the two source docstrings and
Method-to-Code cards, but its equivalent-claim coverage was incomplete. HP-6a
restart is BLOCKED, not PASS. Runtime code and all affected suites are green;
the remaining blocker is current Runtime Atlas status consistency.

The smallest next step is a bounded HP-6a2 documentation repair: update only
U-RT-06/U-RT-08 timing/A/B gap text, broaden the semantic stale assertion to
include `尚缺`, rerun atlas checks and cross-file consistency, and reuse the E70
test evidence because executable source remains unchanged. Return before HP-6b.
