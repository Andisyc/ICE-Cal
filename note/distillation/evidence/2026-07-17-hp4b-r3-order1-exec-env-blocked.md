# HP-4b r3 Order 1 Execution-Environment Block

Date: 2026-07-17

Status: `BLOCKED` before the formal training entrypoint.

## Authorized scope

Execute the eight r3 A/B entries in frozen order, validating each completed
run before starting the next. The frozen fail-fast contract requires the first
non-zero exit to stop the sequence without retry, code/config/workload edits,
or replacement commands.

## Attempted identity

- Identity manifest SHA-256:
  `1f9e447c001476a152852c399d87c2aec57b44453bd5b46904ea6ca6a0de87d7`.
- Frozen cwd: `/private/tmp/unilab-hp4b-f66ab818`.
- Order: 1.
- Route: `legacy`.
- Repetition: 1.
- Raw log:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260716_r3/execution_logs/order_01_legacy.log`.
- Raw log SHA-256:
  `d323a0f8bbe471c5dc6fec957c617d6637ca15a43a0b33b8fdd952a064c5c6e6`.

The raw log records the exact frozen argv, including the E51-repaired workflow
and r3 output path.

## Observed failure

```text
error: Failed to initialize cache at `/Users/chengyuxuan/.cache/uv`
Caused by: failed to open file
`/Users/chengyuxuan/.cache/uv/sdists-v9/.git`: Operation not permitted
```

The outer execution used `/private/tmp/uv-cache`, but `--cache-dir` is an outer
CLI option and is not inherited by the runner's subprocess. The frozen
`argv_prefix` begins with `uv run`; the identity environment does not freeze
`UV_CACHE_DIR`. Consequently the inner command selected uv's default cache and
exited 2 before Python/Hydra started.

E52 had also needed an explicit local dependency-provider environment after a
fresh frozen venv could not download `sentry-sdk`. r3 records neither the
dependency-provider variables nor the cache variable as formal execution
environment identity. Therefore this is a Gate 0B command/environment identity
gap, not a DAgger schema, simulator, collector, learner, or policy failure.

## Isolation facts

- Formal r3 A/B output root remains absent.
- No `legacy_r1/run_manifest.json` exists.
- No scenario dataset, aggregate, checkpoint, metrics, or cleanup artifact was
  created.
- Orders 2-8 did not start.
- No MuJoCo environment, collection, training update, or server mutation ran.
- No retry or command mutation occurred after the failure.

## Decision

HP-4b r3 is `BLOCKED` at order 1. E52's source/archive hashes remain immutable,
but its executable identity is incomplete because the formal dependency/cache
environment was not frozen and exercised through the exact nested subprocess
route.

The next bounded action requires separate authorization: Gate 0B execution-env
repair. Freeze `UV_CACHE_DIR`, dependency-provider selection, no-sync behavior,
and frozen-source import identity in a new manifest; run an exact nested
subprocess preflight that reaches `scripts/train_distill.py --help` or an
equivalent no-training entry boundary; then return before HP-4b execution.
