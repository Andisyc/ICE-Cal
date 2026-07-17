# Persistent Live Run Confirms Learner-Staging Bottleneck

Date: 2026-07-17

Scope: server-side `persistent_async` DAgger run at
`/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_persistent_test01`.

## Evidence

- E87: local mainline merge commit `06d31ad6` combines
  `0abed823` (`High Speed DAgger`) and `f882431b` (HP persistent runtime).
  The exact merged snapshot passed `make test-all`: 1578 passed, 30 skipped,
  256 deselected; Ruff, mypy, and Pyright passed.
- E88: the server `distillation_metrics.json` records
  `execution_mode=persistent_async` for iterations 1-3. All three scenarios
  use collector PID `1127593`; workflow/learner uses PID `1127462`. Scenario
  weight versions are exactly 1, 2, and 3 for outer iterations 1, 2, and 3.
- E89: iteration-2 request totals are approximately 26.64 s across
  `walk_flat`, `static_stand`, and `walk_to_stop`. Workflow stages record
  2.10 s cumulative aggregation, 515.90 s learner batch staging, 144.59 s
  learner forward, 165.44 s learner backward, 12.81 s optimizer step, and
  0.005 s checkpoint save.
- E90: source inspection identifies two active staging costs in
  `src/unilab/algos/torch/distill/offline.py`: `_balanced_batch_indices()`
  rebuilds every label-to-index pool by scanning the full Python label tuple
  on every update; `_indexed_batch()` transfers indices to the dataset device
  and repeatedly calls `indices.detach().cpu()` to reconstruct string labels.

## Facts

- Persistent collector integration is runtime-confirmed, not inferred from
  configuration alone. Worker identity is stable across scenarios and outer
  iterations, while one weight version is shared inside each iteration and
  advances only between iterations.
- Iteration 2 spends about 97% of observed end-to-end time outside scenario
  collection. `learner_batch_staging` is the largest recorded owner stage and
  contributes about 61% of workflow time.
- The current performance owner is therefore the offline learner staging path,
  not persistent worker/env/teacher initialization or cleanup.
- The current implementation mixes CPU Python label planning, CPU-to-device
  index transfer, device `index_select`, and device-to-CPU synchronization in
  one metric. Their individual shares are not yet runtime-confirmed.

## Candidate Optimization Principle

Keep branch-heavy label/index planning on CPU and dense network/tensor work on
GPU, while eliminating repeated full-dataset scans and per-update device
synchronization. The leading candidate is to precompute label index pools once
per cumulative dataset and generate a deterministic batch-index schedule once
or in bounded chunks. This is a hypothesis, not an authorized implementation.

Moving dense learner computation from GPU to CPU is not the proposed principle.
Changing the 8192-update replay budget is also outside this engineering
optimization because it changes replay/training semantics.

## Open Risks

- The current metric does not distinguish label-pool construction, random
  sampling, index transfer, tensor `index_select`, and Python label recovery.
- No microbenchmark has yet measured the attainable staging reduction.
- Iteration 3 learner metrics were incomplete when the server evidence was
  captured because training was still running.
- The run's final cleanup and checkpoint acceptance remain unrecorded here.

## Next

Return control before implementation. This follow-up is named `HP-7`; resume
the main conversation at its A/B/C human gate. Option A is recommended:
`HP-7a`, a no-training microbenchmark over an existing cumulative dataset that
measures the five staging sub-owners separately and compares current per-update
pool reconstruction with a cached-pool/batch-schedule candidate. Options B and
C are design-only and immediate-implementation alternatives respectively; no
option is authorized by this evidence record.
