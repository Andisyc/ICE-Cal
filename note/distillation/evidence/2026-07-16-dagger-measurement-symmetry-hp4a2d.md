# DAgger HP-4a2d Measurement Symmetry Repair

Date: 2026-07-16

Decision: `PASS` for the authorized offline implementation boundary. This does
not rerun Gate 0B, authorize HP-4b, execute MuJoCo/training, or establish a
speedup.

## Owner routes

- HP-4a2d1: the legacy formal collector emits `cold_start`, teacher/student
  inference, env step, tensor pack, artifact write, and total elapsed. Existing
  integer callbacks without a performance context remain artifact-free.
- HP-4a2d2: `workflow.py` owns cumulative aggregation; `offline.py` owns batch
  staging and checkpoint save; `trainer.py` owns learner forward, backward,
  and optimizer step. Timing spans do not alter loss, replay, batch, or update
  semantics.
- HP-4a2d3: `train_distill.py` closes the persistent service before calling the
  workflow finalizer. The finalizer appends one cleanup record, reloads the
  atomic metrics artifact, and refreshes manifest path/hash/count plus the
  lifecycle report. Persistent reports require a positive worker PID and a
  resource-counter mapping; malformed reports fail closed.

Legacy records per-request cleanup ownership in its final report because it
has no resident service. Persistent cleanup identity uses the final iteration's
input checkpoint and exact activated weight version.

## Evidence

- HP-4a2d1 focused gate: `238 passed` and Ruff pass.
- HP-4a2d2 tiny learner/owner gate: `25 passed` and Ruff pass.
- Final affected command:
  `uv run pytest tests/algos/test_distill_performance.py
  tests/algos/test_distill_workflow.py
  tests/algos/test_g1_distillation_contract.py
  tests/scripts/test_train_scripts.py -q`
- Result: `312 passed, 8 skipped, 5 warnings in 7.53s`. Warnings are existing
  torch zero-element initialization warnings in five distillation fixtures.
- Legacy and persistent workflow tests reload the cleanup record and verify
  manifest record count/hash. The persistent malformed-report test rejects the
  report before run-artifact access.

## Remaining boundary

Gate 0B remains `BLOCKED` by its earlier E45 result. Measurement symmetry is
now repaired, but an immutable source bundle has not been frozen and Gate 0B
has not been rerun. The next action requires separate authorization. The active
server process was not inspected, instrumented, stopped, or modified.
