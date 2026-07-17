# HP-6b make test-all Blocked at Repository Ruff Gate

Date: 2026-07-17

Evidence ID: E72

Status: BLOCKED at the first Makefile subtarget. The production gate did not
reach type checking or pytest coverage.

## Authorized Command

From `/private/tmp/unilab-dagger-mainline`:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache \
UV_PROJECT_ENVIRONMENT=/private/tmp/unilab-dagger-mainline/.venv \
make test-all
```

The command used shared-memory permission because the repository non-slow suite
contains spawned IPC/runtime tests.

## Observed Stage Results

1. `uv run ruff format`: completed and reported 57 files reformatted plus 420
   unchanged.
2. `uv run ruff check --fix`: found 17 errors, automatically fixed 15, then
   stopped on 2 F841 errors for which safe fixes were unavailable.
3. `mypy src/unilab`: not started.
4. `pyright`: not started.
5. `pytest -m "not slow" --cov=src/unilab`: not started.

Make exited 2 at the `format` target.

## First Owner Boundary

Both remaining errors are in
`scripts/deploy/check_robojudo_unilab_section8_runtime_torque.py`:

- line 381: local `last_action` is assigned but never used;
- line 382: local `gait_phase` is assigned but never used.

The same names are legitimately used inside two helper rollout functions at
earlier lines; only the two assignments in `main()` are dead. This is an S0
repository lint defect in a RoboJudo/UniLab diagnostic script, not a DAgger
runtime failure.

## Automatic Worktree Mutation

The authorized Makefile gate itself runs formatter and safe lint fixes.
Consequently, 57 Python files were mechanically reformatted and 15 lint issues
were automatically fixed before the failure. The current tracked diff spans 65
files cumulatively. These changes remain in the isolated worktree; no revert or
manual repair was authorized in E72.

The formatter-expanded diff includes files outside the DAgger integration
surface. It must be reviewed as mechanical repository baseline work before any
commit decision. Do not misattribute it to the DAgger feature.

## Evidence Boundary

- E70 remains valid focused evidence: owner probe 10/10, 537 passed, 24 skipped,
  and targeted Ruff before the full sweep.
- E72 does not prove mypy, pyright, full non-slow tests, coverage, slow tests,
  S4 physics, policy quality, or repository production readiness.
- Persistent remains OFF-default and E67 remains `NO_STABLE_SPEEDUP`.

## Decision

HP-6b is BLOCKED. Per the authorized stop condition, the two dead assignments
were not removed and later Makefile targets were not run independently.

The smallest next step is a separately authorized repository-lint repair:
remove only the two unused `main()` assignments, inspect the formatter/auto-fix
diff for semantic changes, and rerun exact `make test-all` from the beginning.
Contract activation, default-on, commit, push, and PR remain closed.
