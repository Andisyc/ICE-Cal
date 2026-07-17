# FT-0 Formal Identity Owner Implementation

Date: 2026-07-17

## Scope

Local owner implementation and deterministic contract tests only. No SSH,
Hydra compose, GPU query, server artifact materialization, environment
construction, collection, learner update, checkpoint, or formal training ran.

## Owner

`src/unilab/algos/torch/distill/formal_identity.py` owns the reviewed formal
lineage, workload, owner-CLI argv/environment, fresh output paths, source and
hard-artifact freeze records, and generated one-shot supervisor/oracle text.
Deploy scripts remain future thin connectors; the HP-7-specific materializer
was not copied into a second training path.

## Contract Evidence

- Formal lineage accepts only original parent iteration 3 and explicitly
  records `r6_sentinel_promoted=false`.
- r6/HP-7 sentinel paths, non-positive workload, non-`persistent_async` mode,
  non-CUDA device, dirty runtime source, missing input, and existing output fail
  closed.
- The argv uses `uv run --no-sync train --algo distill` through the owner CLI.
- Configured update floor and explicit effective updates remain distinct.
- Supervisor enters the frozen repository root, requires every execution output
  absent, owns one telemetry sampler PID, and runs only frozen argv/environment.
- Oracle source compiles, separates preflight/postflight, never launches
  training, hashes frozen inputs, checks HEAD/output identity, and records
  `training_executed=false` during preflight.

## Tests

- RED: initial collection failed because `formal_identity` was absent; later
  supervisor/output assertions failed before their implementations.
- Targeted owner plus prior HP-7 materializer regression: 15 passed.
- Targeted Ruff: PASS.
- Targeted mypy: PASS.

## Decision

The FT-0 owner implementation is PASS. FT-0 remains PARTIAL: the thin deploy
connector, server source/config/artifact resolution, owner-CLI compose,
dependency/GPU snapshots, artifact hashes, and server no-training preflight
remain unconfirmed. FT-1 remains closed.

