# HP-6a Production Readiness Blocked by Stale Runtime Status

Date: 2026-07-17

Evidence ID: E68

Status: BLOCKED at the first read-only review finding. No test command was run.

## Step Contract

HP-6a was authorized as a read-only production-readiness gate: review the
cumulative OFF-default integration against owner/default/lifecycle/lineage
boundaries, then run affected tests, Ruff, and atlas checks. Source repair,
training, `make test-all`, contract activation, default-on, commit, and PR were
explicitly outside scope. The first review or test failure was the stop
condition.

## Observed Contradiction

- `src/unilab/algos/torch/distill/async_runtime.py:10` says reset/resource live
  timing is not connected and legacy/persistent A/B has not run.
- `src/unilab/algos/torch/distill/performance.py:7` says reset/resource live
  timing, MuJoCo timing, and A/B remain absent.
- E61 records the accepted r8 eight-run A/B with complete timing artifacts.
- E65 records accepted two-iteration persistent lifecycle timing and cache/
  cleanup amortization.
- E67 records accepted r10 8/8 execution plus the pre-registered
  `NO_STABLE_SPEEDUP` result.

The source-level module audit status is therefore stale. These are fragile
runtime-owner docstrings, not an obsolete external note, so they can misroute a
future code review and violate the runtime-probing comment/audit-status
contract.

## Classification and Stop

- Finding: Important contract drift; no runtime correctness failure observed.
- Suspected owner boundary: module-level audit status in `async_runtime.py` and
  `performance.py`.
- Root-cause fact: implementation/evidence advanced through E61-E67 while the
  source audit-status text remained at the pre-HP-4 boundary.
- Tests: not run, because the review failure occurred first and the authorized
  step required immediate stop.
- Source changes: none.

HP-6a is BLOCKED, not partially passed. The smallest next step is a separately
bounded source-status repair: update only these owner docstrings to the current
evidence boundary, search the affected runtime modules for equivalent stale
claims, run targeted Ruff plus the atlas/source consistency check, and then
restart HP-6a from its owner review before executing affected tests.
