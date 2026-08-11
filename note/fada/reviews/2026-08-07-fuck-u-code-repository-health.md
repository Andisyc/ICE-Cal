# FADA Code Quality Repository Health Review

Review mode: `repository_health_review`

Overall assessment: `REQUEST_CHANGES`

## Review Summary

- Review scope: FADA Planner-IDM model, source collector, persistent-async runtime, training
  composition, checkpoint lifecycle, and interactive playback integration.
- Static analyzer: local `fuck-u-code` 2.2.2 from `/Users/sss9999/fuck-u-code`.
- Analyzer command: offline `analyze` of the repository with `.venv`, `node_modules`, checkpoints,
  logs, caches, and `.git` excluded; JSON output at `/tmp/fada-fuck-u-code-report.json`.
- Analyzer coverage: 540 analyzed files, 1,131 skipped files; 14 shell files fell back from
  tree-sitter to the regex parser.
- Repository score: 76.18. This is hotspot triage only, not a correctness verdict.
- Focused runtime-independent verification:
  `UV_CACHE_DIR=/tmp/fada_review_uv_cache uv run --no-sync pytest
  tests/algos/test_fada_planner_idm.py tests/algos/test_fada_playback.py
  tests/algos/test_fada_unilab_training.py -q` -> final verification `42 passed in 0.53s`.
- Repository discipline: active. `AGENTS.md` requires contract-first behavior, business rules at the
  owner layer, thin scripts, backend isolation, fail-closed evidence, and risk-local validation.
- Method-completeness evidence: consumed only as contract boundary; this review does not claim that
  simulator instability is caused or solved by code structure.

## Static Hotspot Triage

| Score | File | Static trigger | White-box classification |
|---:|---|---|---|
| 48.86 | `scripts/train_distill.py` | 3,065 code lines, max CC 39, max function 270 reported by analyzer | `code-confirmed-smell`: FADA semantic rules and training lifecycle are owned by the script |
| 52.31 | `fada_collector.py` | max CC 52, max function 331, max 20 parameters | `code-confirmed-smell`: one public function combines several independently changing boundaries |
| 63.41 | `fada_async_runtime.py` | max function 160, max 15 parameters | `correctness-risk`: partial constructor lifecycle has no rollback |
| 59.11 | `interactive_playback.py` | 1,593 code lines, 11 structure issues | mixed: shared playback host is a hotspot; FADA controller/session boundary itself is comparatively narrow |
| 93.70 | `fada_playback.py` | low comment/naming scores only | `intentional-simple`: history owner is small, fail-closed, and independently tested |
| 92.02 | `fada.py` | low comment/naming scores only | `intentional-complexity`: model/tensor owners are separated and covered by focused tests |

Analyzer severities are not reused as P-levels. Every finding below was confirmed from source and
call/test structure.

## Findings

### P0 - Critical

None.

### P1 - High

1. **Unsafe deserialization is enabled on the FADA resume path.**
   - Location: `src/unilab/algos/torch/distill/fada_training.py:648`.
   - `load_fada_checkpoint()` calls `torch.load(..., weights_only=False)` before validating schema or
     architecture. A malicious or substituted checkpoint can execute pickle payloads before the
     fail-closed checks run.
   - The same checkpoint format is already read by `load_fada_policy_checkpoint()` with
     `weights_only=True` at line 673, so the unsafe mode is not justified by the persisted FADA
     payload shape.
   - Impact: arbitrary code execution under the training user's permissions when a resume checkpoint
     is untrusted or replaced.
   - Required repair owner: FADA checkpoint loader. Preserve optimizer restore and schema rejection,
     but make safe loading the sole supported path or explicitly isolate a user-approved legacy
     conversion step outside training.

2. **Persistent worker construction can leak already-created resources after partial failure.**
   - Location: `src/unilab/algos/torch/distill/fada_async_runtime.py:354-409`.
   - Construction opens `SharedWeightSync`, loads resident policies, creates the walking environment,
     and then creates the standing environment. If any later step raises, Python never returns a
     constructed object and `close()` at lines 586-598 cannot run.
   - Impact: leaked shared-memory handles, simulator resources, and model memory; repeated worker
     startup failures can exhaust a training host.
   - Existing tests do not cover this lifecycle. Worker tests instantiate with
     `PersistentFADACollectorWorker.__new__` and assign private fields directly, for example
     `tests/algos/test_fada_unilab_training.py:885-918` and `945-992`.
   - Required repair owner: persistent worker resource materialization. Add rollback for every
     successfully acquired resource and a constructor-failure regression that asserts exact close
     counts.

3. **The formal training script owns long-term FADA business rules instead of only composition.**
   - Location: `scripts/train_distill.py:2453-3143`.
   - The script validates fixed v005 ratios, validates artifact provenance, owns quality-batch
     construction, runs the persistent learner loop, runs the legacy learner loop, and persists
     checkpoints. These are method/training owners, not CLI assembly.
   - This violates the active `AGENTS.md` rule that `scripts/` assemble flows and long-lived rules stay
     at the owner layer. It creates `Divergent Change`: adding a scenario, changing checkpoint
     evidence, changing async lifecycle, or changing replay policy all modify the same script.
   - Impact: semantic drift between `fada_training.py`, `fada_async_runtime.py`, config defaults, and
     the script; tests must import a 3,273-line entrypoint to exercise local FADA rules.
   - Required repair owner: move the FADA use-case and validation boundary into the existing FADA
     training/runtime modules. Keep `main()` responsible only for Hydra composition and dispatch.

### P2 - Medium

4. **The collector interface is a 20-parameter Data Clump and the loop combines too many lifecycle decisions.**
   - Location: `src/unilab/algos/torch/distill/fada_collector.py:341-671`.
   - `collect_fada_source_windows()` owns reset compatibility, observation projection, command forcing,
     walking/standing Oracle selection, same-state shadow transaction, rollout action selection,
     episode reset, temporal scenario scheduling, window admission, history mutation, and result
     aggregation.
   - The function's 20 keyword parameters recur as a configuration bundle in the persistent worker
     and legacy script call sites. This is a real parameter-object admission condition, not a request
     to split by line count.
   - Impact: a new transition such as standing-to-walking or turn transitions must modify the central
     loop and widen its interface, increasing regression surface.
   - Recommended boundary: one validated collection specification/value object plus a separate
     scenario schedule owner; retain the current pure window builders and one rollout coordinator.

5. **Worker tests depend on private state assembly and therefore do not test the public lifecycle.**
   - Location: `tests/algos/test_fada_unilab_training.py:881-1139`.
   - Multiple tests bypass `__init__` with `__new__`, then assign `cfg`, environments, teachers,
     weight sync, and source allocations directly. This is `Inappropriate Intimacy` and makes the
     Module Test Card unable to state a public construction input/output and independent cleanup
     oracle.
   - Impact: tests can pass while constructor ordering, resource rollback, dependency injection, or
     required fields are broken.
   - Recommended boundary: test a public worker factory with injected lightweight resource factories;
     independently assert successful construction, partial failure rollback, collect, and close.

6. **Velocity-command mutation is duplicated across playback session types and a script fallback.**
   - Locations: `interactive_playback.py:274-301`, `interactive_playback.py:460-482`, and
     `scripts/play_interactive.py:1133-1142`.
   - Shape validation, broadcast, state mutation, and observation refresh are repeated with slightly
     different behavior. The script fallback mutates command state without the same explicit refresh
     contract.
   - Impact: command behavior can drift between RSL-style, off-policy, and fallback playback; this is
     particularly risky for stop/restart and turning diagnosis.
   - Recommended boundary: expose command mutation through the playback session contract only. A
     shared pure validator is sufficient; do not add a pass-through service or framework.

### P3 - Low

7. **Test-dependency compatibility catches can hide an internal `TypeError`.**
   - Location: `src/unilab/visualization/interactive_playback.py:1695-1713`.
   - When injected `deps` are used, any `TypeError` from inside the fake environment factory is treated
     as a signature mismatch and retried with fewer arguments.
   - Production dependencies re-raise immediately, so this is test-only diagnostic loss rather than a
     production failure.
   - Recommendation: adapt fake factories to the public signature or inspect the callable signature at
     the test seam instead of catching execution errors.

## Removal And Iteration Plan

No deletion is safe now. The large entrypoints and compatibility branches have active consumers.

1. First close P1 checkpoint safety and constructor rollback with focused negative tests.
2. Characterize legacy and persistent FADA training outputs at their current pinch points.
3. Move FADA validation/use-case orchestration out of `train_distill.py` without changing Hydra
   dispatch or checkpoint schema.
4. Introduce the collection specification only after characterization proves the existing parameter
   bundle and scenario schedule.
5. Remove duplicated script-owned rules after all callers use the FADA owner modules.

## Responsibility And Dependency Delta

- Current added responsibility: `train_distill.py` owns FADA semantics, lifecycle, and persistence in
  addition to CLI composition.
- Current duplicated authority: v005 ratios/defaults and command mutation exist in several modules.
- Current caller knowledge: collector callers know projection, command, temporal, Oracle, replay, and
  cold-start primitives individually.
- Remaining hotspot exception: `interactive_playback.py` is a shared multi-algorithm composition
  host; FADA-specific session/controller code does not by itself justify splitting the entire file.

## Annotation And Atlas Delta

- Added `B1/B2/B3` comments to `PersistentFADACollectorWorker.__init__` so the partial acquisition
  sequence is human-visible. No behavior changed.
- Existing important FADA owners already contain Chinese B-blocks: source collector, persistent
  `collect`, training orchestration, playback controller, and playback session.
- Trivial helpers remain unannotated deliberately.
- No repository Code Quality Atlas was found; no Atlas was generated. Raw analyzer facts remain in
  `/tmp/fada-fuck-u-code-report.json`, separate from this judgment report.

## Evidence Boundary

- The 42 focused tests prove current deterministic contracts still pass.
- They do not prove constructor rollback, safe resume deserialization, closed-loop robot stability, or
  long-horizon Planner/IDM policy quality.
- The reported walking, restart, and turning failures still require the separate runtime differential
  probe; this maintainability review must not claim their physical root cause.
