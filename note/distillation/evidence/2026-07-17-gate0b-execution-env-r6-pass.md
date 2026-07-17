# Gate 0B Execution Environment r6 Pass

Date: 2026-07-17

Status: `PASS`. HP-4b execution remains separately human-gated.

## Scope

Repair E53's exact nested-command environment identity without changing the r3
source archive, frozen source, Hydra config, external assets, workload, run
order, or training semantics. Execute only no-training nested preflights.

## Accepted identity

- Identity:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r6/gate0b_identity_manifest.json`.
- Identity SHA-256:
  `cbf054a84e9b44f4f6a104b8aa458821b5242bc3846731f266444ef88164b778`.
- Reused immutable source bundle SHA-256:
  `f66ab818fc2b013b674e9966597d49c507ce529c1bbee4ebfe4d56036b187191`.
- Frozen cwd: `/private/tmp/unilab-hp4b-f66ab818`.
- New formal output root:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_ab_20260717_gate0b_r6`.
- State: `FROZEN_NOT_EXECUTED`, `execution_authorized=false`.

Two intermediate candidates were rejected before nested execution:

- r4 resolved the venv launcher before `uv pip freeze`, auditing the base
  environment instead of the provider venv.
- r5 correctly captured the provider but retained an r4 output alias and left
  uv/VIRTUAL_ENV resolution implicit.

r6 records both candidates and their rejection reasons. Neither candidate is
an accepted execution identity.

## Frozen execution environment

```text
VIRTUAL_ENV=/private/tmp/unilab-dagger-mainline/.venv
UV_PROJECT_ENVIRONMENT=/private/tmp/unilab-dagger-mainline/.venv
UV_CACHE_DIR=/private/tmp/uv-cache
UV_NO_SYNC=1
PYTHONPATH=/private/tmp/unilab-hp4b-f66ab818/src
UNILAB_DISTILL_PROGRESS=1
```

Dependency provider facts:

- Venv launcher: `/private/tmp/unilab-dagger-mainline/.venv/bin/python3`.
- Resolved Python binary SHA-256:
  `bd6306515305e7505d98dab49007f0196607878b0e4bf9d1f762be2a58804862`.
- Frozen package count: 171.
- `uv pip freeze` snapshot SHA-256:
  `dfa668fddd9b47b7b49c1f63013645420f68ecc9d7554af51450d046a19736bd`.
- Absolute uv launcher: `/Users/chengyuxuan/.local/bin/uv`.
- uv version: `uv 0.11.24 (5e04460c0 2026-06-23 aarch64-apple-darwin)`.
- uv binary SHA-256:
  `60683d72e19df835ddea6bdf3cf1767958abc454de4195ae1cd463d6274a96d3`.

## Exact nested preflight

Preflight artifact:

`/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r6/execution_env_preflight.json`

SHA-256:
`b135a94c7f0a513adaa8002687bb9efb70cbabf24f08009734524097216b92f4`.

1. Nested import used the same absolute uv launcher and frozen environment.
   It exited 0 and observed:

```text
python=/private/tmp/unilab-dagger-mainline/.venv/bin/python3
prefix=/private/tmp/unilab-dagger-mainline/.venv
unilab=/private/tmp/unilab-hp4b-f66ab818/src/unilab/__init__.py
```

2. The exact no-training entrypoint was:

```text
/Users/chengyuxuan/.local/bin/uv run python scripts/train_distill.py --help
```

It exited 0 from the frozen cwd. The formal output root remained absent.

3. The live provider freeze exactly matched the stored 171-package snapshot.
No default user uv cache was accessed.

## Decision

Gate 0B execution-environment repair passes. r6 now freezes the executable
source, uv engine, dependency provider, package snapshot, cache, no-sync mode,
frozen import route, assets, workload, output identity, and run order strongly
enough for a separately authorized HP-4b execution.

No Hydra training, simulator, collection, aggregation, learner update,
checkpoint, metrics, cleanup, or A/B output ran. Control returns before HP-4b.
