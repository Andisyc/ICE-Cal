# HP-7c3 Bounded Persistent Workflow PASS

Date: 2026-07-17

## Scope

One frozen `persistent_async` fork workflow from r6. This is bounded live
integration evidence for the immutable label-pool cache, not a legacy/cached
A/B, promotion trial, or policy-quality experiment.

## Identity And Acceptance

- Oracle accepted with no failures.
- Freeze SHA-256:
  `dbcdf340b593880a02c7fada5bf88bd64b10fd04c09d4976fc60b1a2ef851adc`.
- One completed DAgger iteration, input weight version 1, 12,320 learner
  updates, 853,504 aggregate rows, checkpoint persisted, and cleanup complete.
- Metrics contain 28 successful records under one outer iteration, one weight
  version, and `persistent_async`; oracle validation covers scenario,
  checkpoint, metrics, cleanup, and telemetry artifact contracts.

## Timing

- Wall time: `368.38 s`; process exit status 0.
- Learner batch staging: `34.3355 s`, `0.00278697 s/update`, and `9.3207%` of
  wall time.
- Learner backward: `167.6141 s` (`45.50%` of wall time).
- Learner forward: `131.3816 s` (`35.66%` of wall time).
- Optimizer step: `12.9387 s` (`3.51%` of wall time).
- The four learner stages sum to about `346.27 s`, or `94.0%` of wall time.
- Cumulative aggregation is `3.0223 s`; all three scenario records together
  sum to about `1.8772 s`; cleanup is `0.2784 s`.
- Throughput is `33.4437 updates/s`.

Relative to E92's current-path staging (`31.8345 s / 512`), the bounded run's
per-update staging is about `22.3x` lower. This is consistent with the cache
removing repeated pool construction, but it crosses workload/warmup/timing
boundaries and is not a formal A/B or end-to-end speedup claim.

## Memory Limitation

`time -v` records maximum resident memory `1,901,596 KiB`. The NVIDIA CSV has
7,316 samples and reports a peak `18,264 MiB`, but it contains five GPU process
IDs while the workflow metrics contain only two worker IDs. Therefore the GPU
mean/peak are host-level observations contaminated by unrelated processes and
must not be attributed to this workflow. This limitation does not invalidate
functional or timing acceptance, but it forbids a workflow-specific GPU-memory
claim.

## Decision

HP-7 implementation, production-path wiring, and one bounded persistent live
integration are complete. The measured bottleneck has shifted from repeated
label-pool construction to learner forward/backward. Close HP-7 without another
run. Persistent remains legacy/OFF by default; no end-to-end speedup,
policy-quality, promotion, or default-on conclusion is authorized.

