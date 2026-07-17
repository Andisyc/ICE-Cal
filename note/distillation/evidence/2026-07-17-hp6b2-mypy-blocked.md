# HP-6b2 Full Rerun Blocked at Mypy

Date: 2026-07-17

Evidence ID: E74

Status: BLOCKED at the repository type gate.

## Mechanical Diff Review

Compared every Python file under `scripts/`, `src/`, and `tests/` against the
r8 frozen source `/private/tmp/unilab-hp4b-ea1d4f7a`, stripping docstrings
before AST comparison:

- 429 files are AST-equivalent.
- One file differs: the section-8 diagnostic repaired by E73.
- Its AST diff contains exactly the two removed `main()` assignments.
- No frozen counterpart is missing and no file has a parse error.

Therefore E72's 57-file formatting and 15 safe fixes leave no unexplained
executable AST change. E73 is the only intentional executable delta after the
frozen source.

## Full Rerun

Exact command:

```bash
UV_CACHE_DIR=/private/tmp/uv-cache \
UV_PROJECT_ENVIRONMENT=/private/tmp/unilab-dagger-mainline/.venv \
make test-all
```

Observed stages:

1. Ruff format: 477 files unchanged.
2. Ruff check/fix: all checks passed.
3. Mypy: 20 errors in 8 files; Make exits 2.
4. Pyright: not started.
5. Non-slow pytest with coverage: not started.

## Type-Owner Classification

Branch/runtime-integration surface: 7 errors in 4 files.

- `collector.py:373`: two optional/indexed-assignment errors.
- `async_runtime.py:240`: `Any` returned as `DaggerCollectResult`.
- `workflow.py:650,944,1023`: bool/string inference, optional schema version,
  and updater result union conversion.
- `g1_persistent_worker.py:64`: string passed to the SAC literal contract.

Unchanged-from-HEAD repository baseline: 13 errors in 4 files.

- `models.py`: four `no-any-return` errors.
- `playback.py`: one routing literal error.
- `data.py`: seven optional/index errors.
- `g1/joystick.py`: one `no-any-return` error.

AST comparison against HEAD confirms the first four files are new/changed on
this branch and the latter four are AST-identical to HEAD. This classification
does not waive baseline errors: `make test-all` requires all 20 to pass.

## Coverage Boundary

- S0 format/lint: PASS.
- S0 static typing: BLOCKED at mypy.
- Pyright: unconfirmed.
- S1/S2/S3 full non-slow suite and coverage: unconfirmed in HP-6b.
- E70 focused evidence remains 537 passed/24 skipped.
- S4 physical-policy quality remains separate and E28 is still BLOCKED.

## Decision

HP-6b remains BLOCKED. No type repair or later Makefile subtarget was executed
after the first failure.

The next repair must be split by owner: first the 7 branch-owned DAgger runtime
errors, then the 13 unchanged repository-baseline errors. Each step needs
targeted mypy and local tests before another exact `make test-all`. Contract
activation, default-on, commit, push, and PR remain closed.
