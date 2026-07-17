# FT-1 Acceptance Oracle v2 Implementation

Date: 2026-07-17

## Scope

Implement a standalone, read-only v2 postflight oracle for the already executed
r2 formal run. No training, artifact repair, resume, retry, deletion, or server
validation occurred in this step.

## Added Acceptance Boundaries

- Frozen runtime source and hard-artifact byte identities.
- Dependency/import and physical GPU identity.
- Two manifest iterations and exact update schedule.
- Parent checkpoint -> iteration 1 checkpoint -> iteration 2 checkpoint hash
  lineage and monotonic input weight versions.
- Scenario order, sample count, shared input weight version, and persistent
  collector worker identity per iteration.
- Aggregate/checkpoint existence and manifest SHA-256 for every iteration.
- Required learner/checkpoint metrics per iteration; scenario metric
  execution-mode, weight-version, and checkpoint identity.
- Exactly one complete cleanup metric plus complete manifest cleanup state.
- Metrics hash/count and nonempty log/time/GPU telemetry.
- Final checkpoint path and SHA-256 in the v2 result.

HEAD drift caused only by adding the oracle repair is recorded as a warning,
while every frozen runtime byte remains hard-gated. This permits revalidation
without mutating the r2 freeze or pretending the new oracle existed before the
run.

## Verification

- Complete two-iteration artifact-chain fixture: accepted.
- Lineage, cleanup, dependency, and GPU drift fixture: rejected.
- Missing final checkpoint and cleanup metric fixture: rejected.
- Focused postflight/formal/workflow suite: 41 passed.
- Ruff and mypy: PASS.

## Decision

Oracle v2 implementation PASS locally. FT-1 full artifact acceptance remains
PARTIAL until this oracle evaluates the existing r2 artifacts. Training must
not be rerun.

