# FT-0 Deploy Connector Integration

Date: 2026-07-17

## Scope

Local connector implementation and file-level integration tests only. No SSH,
server materialization, training, environment, collector, learner, checkpoint,
FT-1, or RT-10 execution occurred.

## Owner Boundary

- `formal_identity.py` remains the semantic owner for formal lineage, workload,
  command, outputs, freeze, supervisor, and oracle.
- `materialize_formal_dagger_gate0.py` is a thin deploy connector. It reads a
  reviewed JSON spec, observes Git/Hydra/dependency/GPU facts, writes generated
  artifacts, and invokes only the oracle `--preflight` mode.
- The connector never invokes the frozen training argv. The supervisor remains
  generated but unexecuted.

## Confirmed Contracts

- Seven hard artifact identities are mandatory: parent manifest, parent
  checkpoint, parent aggregate, walk/stand teachers, and walk/stand datasets.
- Owner-CLI Hydra compose is generated from the frozen formal argv with
  `--cfg job --resolve`; empty output, nonzero exit, or stderr fails closed.
- Runtime cleanliness uses `git status --porcelain --untracked-files=all` over
  the runtime scope, so untracked owner files cannot appear clean.
- Dependency/import and GPU observations are frozen alongside the compose,
  source, artifact, command, and output identities.
- Freeze, compose, supervisor, oracle, and preflight paths refuse overwrite.
- A temporary Git-repository integration fixture materializes an accepted
  preflight while leaving the formal run directory absent and reporting
  `training_executed=false`.

## Verification

- Connector RED: both tests initially failed because the connector did not
  exist. The untracked-runtime test later failed against `git diff`-only
  cleanliness and passed after the owner used porcelain status.
- Connector/owner/HP-7 regression: 18 passed.
- Targeted Ruff: PASS.
- Targeted mypy: PASS.

## Decision

FT-0 deploy connector integration is PASS locally. Full FT-0 remains PARTIAL
until a reviewed formal spec fixes the formal outer-iteration/workload/output
identity and an authenticated server session runs exactly one no-training
materialization/preflight. FT-1 remains closed.

