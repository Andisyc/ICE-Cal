# FADA Request-Scoped Collector Stabilization

## Admission card

- Observed failures: two collector SIGSEGV incidents at NumPy validation victims and one
  impossible Python unpack symptom after long FADA collection.
- Frozen probes: no more validation prints, sentinel checks, environment-lifetime variants, or
  full-training localization runs because those paths no longer change the repair decision.
- Victim/trigger: Python/NumPy window validation exposes damage after repeated native rollout work.
- Owner: the DAgger collector process owns simulator, Oracle, CUDA, environment, and cleanup state
  for one collection request.
- Surviving corrupter classes: native state retained across requests, CUDA/native allocator
  lifetime, and incomplete cleanup after a request. The exact low-level writer remains unknown.
- Common invariant: native simulator and accelerator state from one FADA request must never be
  observable by the next request.
- Closure: stabilization. Exact-writer capture would not change the request-process boundary.

## Accepted repair

1. Keep the generic DAgger default persistent for compatibility, but allow the composition root to
   select a request-scoped worker lifecycle.
2. Select request scope only for FADA: start one spawned collector for one iteration request,
   receive and validate its result, request cooperative shutdown, join it, inspect its exit status,
   close process/error handles, and only then allow another request.
3. Preserve the shared CPU weight publication boundary. Every fresh child attaches to the same
   versioned `SharedWeightSync` and validates the expected weight version.
4. Persist request ID, scenario, iteration, checkpoint path, weight version, and producer PID inside
   the CPU-only atomic source artifact. The learner rejects incomplete or mismatched identity before
   replay mutation.
5. Retain the existing exact environment/backend snapshot contract and its exception round-trip
   tests. This repair does not change Reward, observations, actions, windows, curriculum, domain
   randomization, Oracle lineage, optimizer updates, or checkpoint semantics.

## Offline proof and stop boundary

- RED: a request-lifecycle runner test fails because the current runner has no request ownership
  mode and reuses one PID.
- GREEN: two sequential requests have distinct worker PIDs, each worker reports cleanup, and the
  runner retains no process handle after either request.
- Negative path: a valid result is rejected when worker cleanup fails; stale artifact identity is
  rejected before learner replay.
- Run the affected algorithm/base suites plus Ruff, formatting, Pyright, compileall, and diff checks.
- Offline evidence fixes the collector owner-boundary class only. One bounded official server
  transaction remains the sole live confirmation; no repeated diagnostic campaign is authorized.

