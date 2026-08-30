# Off-policy Lifecycle Owners Implementation Plan

> **For agentic workers:** Execute inline in one authorized transaction. Steps
> use checkbox syntax for evidence tracking; no branch, commit, live run, or
> external operation is authorized.

**Goal:** Replace two active off-policy god functions with explicit per-run
lifecycle owners while preserving every public and runtime-visible behavior.

**Architecture:** `OffPolicyCollectorSession` owns one collector subprocess
lifecycle behind the unchanged entrypoint. `DoubleBufferTrainingSession` owns
one learner-run resource and iteration lifecycle behind the unchanged runner.

**Tech Stack:** Python 3, PyTorch, NumPy, multiprocessing, pytest, Ruff,
`uv run`.

**Spec:** `docs/superpowers/specs/2026-08-31-offpolicy-lifecycle-owners-design.md`

## Global constraints

- Preserve Actor/privilege/tensor/replay/queue/update/checkpoint/log/cleanup
  semantics and public signatures.
- Preserve current module monkeypatch seams through explicit dependency
  records constructed at the existing composition roots.
- Do not add a protocol, schema, backend branch, normalization, clamp, retry,
  fallback, or live operation.
- Preserve the dirty worktree and use only `uv run`.

### Task 1: Add architecture and compatibility fitness tests

**Files:**
- Create: `tests/algos/test_offpolicy_lifecycle_owner_boundaries.py`

**Interfaces:**
- Produces: source-level and delegation checks for the two lifecycle owners.

- [ ] Assert `DoubleBufferOffPolicyRunner.learn` delegates to
  `DoubleBufferTrainingSession` and no longer contains replay/training-loop
  implementation.
- [ ] Assert `off_policy_collector_fn` retains its current named parameters and
  delegates through `OffPolicyCollectorSpec` and `OffPolicyCollectorSession`.
- [ ] Assert the session classes expose explicit preparation, iteration/step,
  metrics/finalization, and cleanup phases.
- [ ] Run the test and confirm RED is caused only by the absent owners.

### Task 2: Extract the collector request and session

**Files:**
- Create: `src/unilab/algos/torch/offpolicy/collector_session.py`
- Modify: `src/unilab/algos/torch/offpolicy/worker.py`
- Test: `tests/algos/test_offpolicy_worker.py`
- Test: `tests/algos/test_offpolicy_runtime.py`
- Test: `tests/algos/test_offpolicy_bootstrap_contract.py`

**Interfaces:**
- Consumes: the existing `off_policy_collector_fn` values and worker decision
  helpers.
- Produces: `OffPolicyCollectorSpec`, `OffPolicyCollectorDependencies`, and
  `OffPolicyCollectorSession.run() -> None`.

- [ ] Pin the exact existing entrypoint parameter names and privilege/replay
  behavior in tests.
- [ ] Move mutable child-process state into `OffPolicyCollectorSession`.
- [ ] Split initialization, action selection, environment/replay step, sync,
  metrics publication, and close into session phases.
- [ ] Keep `off_policy_collector_fn` and `_run_collector` as thin adapters and
  preserve error propagation to the Async Runner wrapper.
- [ ] Run worker, privileged-input, terminal replay, pack-request, and runtime
  tests until GREEN.

### Task 3: Extract the double-buffer learner run session

**Files:**
- Create: `src/unilab/algos/torch/offpolicy/double_buffer_session.py`
- Modify: `src/unilab/algos/torch/offpolicy/double_buffer_runner.py`
- Test: `tests/algos/test_offpolicy_double_buffer_runner.py`
- Test: `tests/algos/test_offpolicy_runner_unit.py`
- Test: `tests/ipc/test_replay_pipeline_double_buffer.py`

**Interfaces:**
- Consumes: one configured `DoubleBufferOffPolicyRunner` and frozen run
  options/dependencies.
- Produces: `DoubleBufferTrainingSession.run() -> None`.

- [ ] Pin public delegation, collector kwargs, checkpoint cadence/final save,
  trace events, sync tokens, and dead-collector cleanup in tests.
- [ ] Move run-local resources/counters into the session and retain runner
  configuration and existing narrow helper ownership.
- [ ] Split memory/resource preparation, logger/queue creation, collector
  start, readiness, one iteration, checkpoint/log publication, normal
  finalization, and failure cleanup.
- [ ] Preserve factory seams by constructing the dependency record from the
  runner module globals.
- [ ] Run double-buffer runner, runner unit, checkpoint adapter, and replay
  pipeline tests until GREEN.

### Task 4: Close the complete engineering unit

**Files:**
- Create: final review and one-shot receipts under `docs/superpowers/`.

**Interfaces:**
- Consumes: the complete diff and test evidence.
- Produces: validated `FINAL_GATE_PASS` and `COMPLETE` receipts.

- [ ] Run architecture tests, the complete off-policy/IPC affected suite,
  scoped Ruff, compileall, and import checks.
- [ ] Run repository pytest and classify only demonstrated pre-existing
  failures.
- [ ] Recount file and largest-function sizes; reject a pass if either new
  session is only a renamed god method or duplicates lifecycle ownership.
- [ ] Validate the final `code-review-expert` maintainability delta and
  `one-shot-execution` completion receipts.
