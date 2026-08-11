# UniLab FADA Persistent Async Architecture Evidence

Date: 2026-08-05

## Accepted architecture

| Boundary | Owner | Contract |
|---|---|---|
| Environment and rollout lifecycle | spawned FADA collector | Created once and closed by the worker |
| Final Oracle | spawned FADA collector | Resident across iteration requests |
| Intermediate Oracles | spawned FADA collector | Loaded one at a time to bound accelerator memory |
| Planner-IDM rollout copy | spawned FADA collector | Updated only through versioned `SharedWeightSync` |
| Replay and optimizer state | parent learner | Mutated only after a validated artifact returns |
| Checkpoint | parent learner | Paired Planner/IDM/optimizer identity, atomically saved |

Each request represents one complete DAgger collection iteration. The worker first collects the
optimal/current-policy source, then the paper-source allocations, and writes one architecture-bound
causal-window artifact. The parent validates the request identity, weight version, artifact schema,
architecture, and sample count before replay insertion. It then performs the ordered IDM and fixed-IDM
Planner updates and publishes the next checkpoint version.

This deliberately preserves the DAgger iteration barrier. `persistent_async` means environment and
rollout resources live in a persistent spawned process; it does not permit collection using stale
Planner-IDM weights while the parent updates the next version.

## Local evidence

- `tests/algos/test_fada_unilab_training.py`: 14 passed.
- FADA source artifact round-trip rejects architecture drift.
- FADA worker test returns current-policy plus intermediate-Oracle rows under one observed weight version.
- Parent async-route test publishes a new version after each learner update and closes its runtime.
- Ruff passes for all touched FADA/runtime files.
- Pyright passes for `fada_async_runtime.py` and `fada_training.py`; the monolithic entry script retains pre-existing unrelated diagnostics.

No simulation or training was launched for this architecture change. No server access is part of this evidence.
