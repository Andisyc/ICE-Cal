# Formal DAgger Auto Output Identity

Status: AOI-1/AOI-2 local implementation PASS; AOI-3 is a separate user-controlled Gate 0 decision.
Date: 2026-07-20.

## Decision

The human supplies a semantic `run_name`, not filesystem timestamp strings.
During no-training Gate 0 materialization, the formal identity owner resolves
one local-time, lexically sortable stem:

```text
YYYYMMDD-HHMMSS_<run_name>
```

It derives both the workflow `run_dir` and the formal fresh `artifact_dir`
from repository-relative default roots, then freezes the resolved absolute
paths in the command, freeze, compose, supervisor, and preflight result. The
same supervisor never regenerates a timestamp; it remains one-shot and fails
closed if any frozen output exists.

## Scope And Non-Scope

Scope:

- Add the formal owner-local `run_name` to output-path resolution.
- Keep explicit `run_dir` / `artifact_dir` specs backward compatible.
- Reject ambiguous specs that provide both a generated-name request and manual
  output paths.
- Return the resolved paths from the deploy materializer for human control.

Non-scope:

- DAgger semantics, teacher/data identity, replay schedule, CUDA device,
  batch/sample/env counts, OOM mitigation, logging policy, or server training.
- Automatic retries, output deletion, resume, or timestamp generation in the
  supervisor.

## Ownership

| Object | Owner | Consumer | Forbidden behavior |
| --- | --- | --- | --- |
| Auto output identity | `formal_identity.py` | Gate 0 connector | script-owned path policy or per-launch timestamp changes |
| Input spec parsing | `materialize_formal_dagger_gate0.py` | formal owner | inventing a second naming rule |
| Frozen paths | formal freeze/supervisor/oracle | server execution | overwrite, resume, or retry under the same identity |

## Step Map

### AOI-1 / 3: Owner and connector contract

- Objective: Resolve a `run_name` into one frozen, time-sorted identity.
- Files: `formal_identity.py`, deploy materializer, their focused tests.
- Evidence: deterministic fixed-clock output, manual-path compatibility,
  ambiguity and invalid-name failures, materializer result paths.
- Stop: focused contract/connector tests pass without training.

### AOI-2 / 3: Governance and Architecture refresh

- Objective: Record the control surface without changing the Concept Figure.
- Files: this plan, current checklist, task canvas, evidence ledger, formal
  owner cards in Runtime and Method-to-Code Atlas.
- Evidence: docs and Atlas checks pass.
- Stop: current state distinguishes `run_name` request from frozen paths.

### AOI-3 / 3: New fresh OOM-r2 identity

- Objective: separately choose workload/resource overrides, materialize a new
  identity, and run no-training Gate 0.
- Non-scope: AOI-1/AOI-2 do not create r2 or contact the server.
- Authority: user confirmation required after AOI-1/AOI-2.

## Acceptance Matrix

| Item | Owner | Evidence | Status |
| --- | --- | --- | --- |
| Deterministic generated stem | formal identity | S1 fixed-clock unit test | PASS, E109 |
| Manual identity compatibility | connector | S1 parsing/materialization test | PASS, E109 |
| Frozen resolved output paths | connector | S2 no-training materialization fixture | PASS, E109 |
| Atlas and docs agreement | governance/Atlas | S0 docs + Atlas check | PASS, E109 |
| Authenticated r2 Gate 0 | server connector | S3 no-training preflight | PASS, E111 |

## AOI-1/AOI-2 Result

`FormalDaggerAutoOutputIdentity` is the only new path-policy owner. With a
fixed Gate 0 clock, it deterministically resolves one
`YYYYMMDD-HHMMSS_<run_name>` stem. The deploy connector copies that resolved
identity into the ordinary formal specification, reports the resulting paths,
and records `auto_output_identity` in the immutable freeze document.

The owner creates no directories and starts no commands. The generated
supervisor has no clock logic: it consumes the already-frozen argv and remains
one-shot/fail-closed. Existing reviewed specifications that explicitly contain
`run_dir` and `artifact_dir` are still accepted. A specification that mixes
`run_name` with either manual output path is rejected.

Local evidence is E109. It is not an authenticated server Gate 0 and does not
select an OOM mitigation, workload, or new formal run.

E110 now records one approved local fresh-r2 spec using this control surface.
It freezes `run_name=g1_walk_stand_formal_fresh_8iter_oom_r2` without manual
output paths. E111 records the separately authorized Gate 0 PASS: one server
timestamp identity was materialized and preflight accepted without training.
