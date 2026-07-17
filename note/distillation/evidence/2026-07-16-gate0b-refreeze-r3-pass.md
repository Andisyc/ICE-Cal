# Gate 0B Refreeze r3 Pass

Date: 2026-07-16

Status: `PASS`. HP-4b execution remains separately human-gated.

## Scope

Refreeze the exact E51-repaired source/config identity, prepare a new immutable
cwd, and prove frozen-cwd build/import/source/assets/XML/compose/test readiness.
No HP-4b A/B run, collection, training, server mutation, r2 mutation, partial
run reuse, or performance conclusion is in scope.

## Frozen identity

- Branch: `codex/dagger-mainline-runtime`.
- Base commit: `601a2e4013368423540554a351062b012b4c83ce` plus the captured dirty
  Git-visible source inventory.
- Inventory: `git ls-files -co --exclude-standard`, 1244 files.
- Bundle:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260716_r3/unilab_dagger_source_snapshot.tar.gz`.
- Bundle SHA-256:
  `f66ab818fc2b013b674e9966597d49c507ce529c1bbee4ebfe4d56036b187191`.
- Source manifest SHA-256:
  `69ce41e3f34a9225697eaa26156261fd1db6becf7297722a224cf7add7e9f87b`.
- Identity manifest:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260716_r3/gate0b_identity_manifest.json`.
- Identity SHA-256:
  `1f9e447c001476a152852c399d87c2aec57b44453bd5b46904ea6ca6a0de87d7`.
- Frozen cwd: `/private/tmp/unilab-hp4b-f66ab818`.
- Formal output root:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_ab_20260716_gate0b_r3`.
- Identity state: `FROZEN_NOT_EXECUTED`, `execution_authorized=false`.

The source bundle was generated twice independently. Both runs produced the
same bundle, source-manifest hash, size, file count, branch, and base commit.
The identity explicitly supersedes r2 because E51 changed workflow source.

## Static identity facts

- Required build inputs `pyproject.toml`, `README.md`, `LICENSE`, `uv.lock`,
  and `src/unilab` are present.
- All seven parent/checkpoint/teacher/dataset asset hashes match r2's frozen
  workload identity.
- Workload remains one outer iteration, 128 rows per scenario, 8 updates,
  4 envs, seed 1, CPU, and scenario order
  `walk_flat/static_stand/walk_to_stop` with quotas `0.5/0.25/0.25`.
- Eight run-order entries remain balanced and unchanged.
- Shared Hydra config SHA-256 remains
  `d6e047f43e03de0d13af32823f09a7b538dba21672540e455e51f61f869b2000`.
- Route-specific hashes are
  `84d1b3e29c823bbff4b27f8e7048d621b6aaa9a7db4dbd10042a562c94dd4101`
  for legacy and
  `39a3e2c997c6a5f84430989a26311dbb1fc0c5237141cea3d7caff673f6de4a4`
  for persistent_async. The only differences are execution mode and unique
  run directory.

## Frozen-cwd preflight

`uv build` from the extracted cwd succeeded and produced both sdist and wheel.
The first isolated `uv run` attempt then stopped before import because the new
venv attempted to download `sentry-sdk` while network access was unavailable.
This was classified as dependency acquisition, not a source defect.

The preflight was rerun without network using the existing verified project
venv only as the dependency provider, with
`PYTHONPATH=/private/tmp/unilab-hp4b-f66ab818/src`. An explicit assertion proved
the loaded package path was:

```text
/private/tmp/unilab-hp4b-f66ab818/src/unilab/__init__.py
```

Observed frozen facts:

```text
source/file_count: 1244
source/all_hashes_match: true
assets/all_hashes_match: true
mujoco/nq: 36
mujoco/nv: 35
mujoco/nu: 29
compose/allowed_diff_only: true
outputs/all_absent: true
```

Frozen-cwd affected suite:

```text
312 passed, 8 skipped, 5 warnings in 7.39s
```

## Decision

Gate 0B r3 passes S0/S2/S3 T-persist/T-oracle and frozen-cwd executable
readiness. The formal A/B output root remains absent, so zero HP-4b runs were
started. Post-freeze governance Markdown records are intentionally outside the
immutable executable archive; no code/config/asset inside r3 is changed.

Control returns to the user. HP-4b may run only after separate authorization,
only from `/private/tmp/unilab-hp4b-f66ab818`, and only against the exact r3
identity and run order.
