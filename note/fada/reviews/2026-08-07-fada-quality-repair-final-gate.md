# FADA Code Quality Repair — Final Gate

Review mode: `final_gate_review`

Outcome: `APPROVE_WITH_FOLLOWUPS`

## Closed Findings

| Original finding | Repair | Behavioral oracle |
|---|---|---|
| P1 unsafe FADA resume deserialization | `load_fada_checkpoint()` now uses `weights_only=True` | resume round-trip plus a spy test proves safe mode is the only call |
| P1 partial worker-construction leak | constructor acquisition is transactional and rolls back unique environments plus shared-weight sync | real-constructor failure test asserts exact close counts |
| P1 script-owned FADA rules | FADA validation, replay selection, learner loops, and checkpoint lifecycle moved to `fada_workflow.py`; the script now injects dependencies and dispatches | legacy and persistent-async workflow tests pass through the script wrapper |
| P2 20-argument collector clump | immutable `FADACollectionSpec` owns projection, command, scenario, transition, shadow, and eligibility settings | all collector scenarios and async/legacy callers pass focused tests |
| P2 private-only worker lifecycle coverage | added a public-constructor rollback test with injected lightweight resource factories | walking environment and shared sync each close once when standing environment creation fails |
| P2 duplicated command mutation | shared finite/broadcast validator plus session-owned mutation; script fallback removed | RSL/off-policy command tests and entrypoint boundary test pass |
| P3 `TypeError` signature guessing in FADA playback | FADA environment factory is called once against its public signature | internal `TypeError` propagates unchanged and call count remains one |

## Compatibility Contract

| Boundary | Before | After | Status |
|---|---|---|---|
| Planner/IDM architecture, losses, replay semantics | existing v004/v005 contract | unchanged | preserved |
| Training mode selection | `legacy` or `persistent_async` | unchanged | preserved |
| Checkpoint schema and optimizer restore | schema v2 payload | unchanged payload; safe loader | preserved and hardened |
| Collector behavior | individual keyword bundle | equivalent immutable specification | internal callers migrated |
| Interactive command behavior | session methods plus raw script fallback | session methods only | intentional fail-closed tightening |
| Training/simulation | no run in this repair | no run | no claim about locomotion quality |

## Verification Evidence

- Focused regression:
  `95 passed in 0.95s` across Planner-IDM, playback, UniLab FADA training, interactive playback,
  and visualization entrypoints.
- Focused Pyright on repaired source boundaries: `0 errors, 0 warnings, 0 informations`.
- Ruff lint and format gate on all touched repair files: passed.
- `git diff --check`: passed.
- Final `fuck-u-code` 2.2.2 scan: 541 files analyzed, repository score `76.23`
  (baseline `76.18`). `scripts/train_distill.py` improved from score `48.87`, 3,056 code lines
  to score `51.30`, 2,456 code lines. The score is triage evidence, not a correctness oracle.
- Raw final analyzer report: `/tmp/fada-fuck-u-code-report-final.json`.

## Follow-ups (Non-blocking)

1. `collect_fada_source_windows()` remains a complex rollout coordinator even though its public
   parameter clump is closed. Extract a scenario schedule only when a new scenario is added and its
   independent input/output contract is known.
2. Several older worker behavior tests still use `__new__` fixtures. Migrate them incrementally to
   the injected public constructor; the new rollback path itself is covered through real construction.
3. `interactive_playback.py` remains a shared multi-algorithm host. Split only with a consumer-backed
   ownership boundary, not solely because of file length.

## Evidence Boundary

This gate approves the maintainability and lifecycle repair. It does not establish that the trained
checkpoint walks stably, restarts safely, or turns reliably; those claims require simulator/runtime
probing and, if needed, new data or training.
