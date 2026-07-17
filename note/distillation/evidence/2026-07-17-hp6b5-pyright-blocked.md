# E77 — HP-6b5 Final Repository Gate Rerun

Result: **BLOCKED** at Pyright.

Exact command:

```text
UV_CACHE_DIR=/private/tmp/uv-cache \
UV_PROJECT_ENVIRONMENT=/private/tmp/unilab-dagger-mainline/.venv \
make test-all
```

Observed stages:

- Ruff format: 2 files reformatted, 475 unchanged.
- Ruff check/fix: pass.
- mypy: `Success: no issues found in 233 source files`.
- Pyright: 6 errors, all in branch-owned `collector.py`.
- Coverage pytest: not started.

The Pyright errors are two instances where an optional student policy reaches
`_policy_actions` (lines 578 and 895), plus two NumPy expressions involving an
optional transition-command array (reported as four diagnostics at lines
950/952). These are narrower static-flow facts than the repaired E75 mypy gate.

Per the frozen E77 stop condition, no repair was attempted after this new first
failure. A separate collector-owner Pyright repair gate is required before any
new full rerun.
