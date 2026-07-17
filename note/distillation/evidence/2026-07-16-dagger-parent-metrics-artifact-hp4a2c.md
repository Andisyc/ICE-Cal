# DAgger Parent Metrics Artifact HP-4a2c

Date: 2026-07-16
Scope: persistent request identity enrichment, run-local JSON persistence,
completed-iteration resume, and formal OFF-default entrypoint assembly.

## Owner Route

```text
resolved Hydra cfg + every RoleArtifactSpec teacher checkpoint
-> DistillationPerformanceRunContext
-> persistent worker seven-stage observations
-> parent outer-iteration/request/checkpoint/version identity
-> DistillationMetricsRecorder
-> <run_dir>/distillation_metrics.json atomic replace
-> immediate reload validation
-> run_manifest.json latest path/hash/record count
```

`performance.py` owns schema, run context, exact stage order, enrichment,
deduplication, and atomic persistence. `workflow.py` owns parent request facts,
artifact lifecycle, and completed-iteration resume. `scripts/train_distill.py`
only constructs the context for `persistent_async`; `legacy` passes `None`.

## Deterministic Facts

- Required request stages are exactly `weight_sync`, `teacher_inference`,
  `student_inference`, `env_step`, `tensor_pack`, `artifact_write`, and
  `total_elapsed`.
- The full-run teacher identity is the sorted unique SHA-256 set from every
  configured role teacher checkpoint. The resolved config hash retains the
  role-to-teacher mapping.
- The parent hashes the current student checkpoint once per outer iteration and
  uses the same path/hash and activated weight version for every scenario.
- Each scenario write uses atomic replacement and is immediately reloaded. A
  completed resume validates manifest path/hash/count and run signature before
  adding later records; entering the same completed target is idempotent.
- A missing artifact for an already completed persistent run, a modified file,
  identity drift, malformed schema/stage order, or mixed legacy/performance
  context fails closed.
- Legacy workflow tests create no `distillation_metrics.json` and receive no
  performance context. The configured default remains `legacy`.

## Evidence

- Test-first HP-4a2c1 failed at import because the run-context owner did not yet
  exist; after implementation the focused schema suite passed `22 passed`.
- Test-first HP-4a2c2 failed because the workflow did not accept
  `performance_context`; after implementation workflow/performance passed
  `38 passed`.
- Test-first HP-4a2c3 failed because script assembly omitted the context. The
  next probe found and corrected the actual seed owner from root `seed` to
  `algo.seed`; three exact script connector tests then passed.
- Final affected suite: `373 passed in 13.82s` across performance, workflow,
  differential, worker, async/runtime/resources, train script, and config tests.
- Final two-teacher formal connector fixture: `2 passed`; both distinct teacher
  file hashes enter the context.
- Ruff passes for affected source/tests; `uv run python -m py_compile` passes;
  Atlas check reports `runtime_modules=9 method_modules=11 concept_nodes=6`.

## Limitations

- This is S1/S2/S3 deterministic and offline-connectivity evidence. It is not a
  MuJoCo timing run and proves no speedup or policy quality.
- Reset/resource construction and cleanup-final stage observations remain
  outside HP-4a2c.
- A crash after a scenario artifact write but before the iteration manifest
  commit fails closed on resume. HP-4a2c does not add a scenario-level workflow
  transaction protocol.
- Existing completed persistent runs created before this artifact contract
  cannot be silently upgraded; missing metrics fail closed.
- Gate 0B, HP-4b, HP-4c, HP-5, Motrix, and the active server training process
  were not entered or modified.

## Decision

HP-4a2c passes its parent identity, run-local persistence/resume, and formal
OFF-default connector gates. The next performance decision is Gate 0B; it must
freeze immutable A/B inputs and return control before any bounded run.
