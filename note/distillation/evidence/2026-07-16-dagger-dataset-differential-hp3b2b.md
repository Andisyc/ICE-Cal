# DAgger Dataset Differential HP-3b2b Evidence

Date: 2026-07-16

Scope: deterministic role and walk-to-stop collection through the actual
collector/data owners. No subprocess, production factory, or MuJoCo claim.

## Contract

Legacy collectors reset their env when `initial_reset=None`. Persistent
resources reset once and pass `(obs_dict, info_dict)` through `initial_reset`.
Malformed reset handoffs fail closed. The persistent path must not reset twice.

## Differential Facts

For role collection, legacy and persistent outputs match sample count,
student/teacher dimensions, role labels, active intent labels, teacher action
targets, and teacher checkpoint identity.

For `walk_to_stop`, they additionally match scenario labels, teacher-role and
intent schedule, transition ages, and cached teacher actions. Both paths reset
exactly once.

## Evidence

Red: both collectors rejected the missing `initial_reset` argument.

Green focused differential: `3 passed`.

Collector impact group: `17 passed, 66 deselected`.

## Decision

HP-3b2b passes its deterministic semantic differential. Production worker
wiring may proceed. This does not prove nondeterministic MuJoCo equality or
physical behavior.
