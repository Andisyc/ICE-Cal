# DAgger Resource Lifecycle HP-3b2a Evidence

Date: 2026-07-16

Scope: exact cold-path identity, fake teacher/env lifecycle, per-request reset,
counters, and cleanup. This is not a real G1 dataset or MuJoCo claim.

## Identity Contract

The cache key includes task owner, task name, backend, resolved env-config
fingerprint, num envs, teacher checkpoint path/hash, and teacher-spec
fingerprint. Role strings are not cache keys.

## Runtime Facts

- repeated identical walking requests reuse one teacher/env bundle;
- standing creates its own exact bundle;
- changing only `num_envs` creates a distinct bundle;
- changing only the teacher checkpoint hash creates a distinct bundle;
- every request resets command, done, and transition-age state before the
  collector callback sees the env;
- repeated requests do not grow init counters;
- normal and exceptional paths close each initialized teacher/env exactly once.

## Evidence

Red: module import failed because `persistent_resources.py` did not exist.

Green:

```text
2 passed in 0.03s
```

Ruff: `All checks passed!`.

## Decision

HP-3b2a passes its fake lifecycle gate. HP-3b2b may connect this owner to the
existing collector/data semantics and run the required legacy/persistent
dataset differential. Real G1 resources remain unconfirmed.
