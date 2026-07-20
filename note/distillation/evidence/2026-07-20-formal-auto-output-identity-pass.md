# Formal Auto Output Identity Local PASS

Date: 2026-07-20.
Status: local implementation and formal-route fixture PASS; authenticated server Gate 0 not run.

## Scope

The human-approved control surface replaces manual timestamp construction with
a semantic `run_name`. It covers only formal DAgger output identity resolution:

```text
run_name
  -> Gate 0 local-time stem YYYYMMDD-HHMMSS_<run_name>
  -> frozen run_dir and, for fresh mode, artifact_dir
  -> existing formal command/freeze/supervisor/oracle path
```

It does not change DAgger collection/update semantics, replay budget, samples,
batch size, device, teacher/data identity, default execution mode, logging,
OOM behavior, retry, resume, or server execution.

## Owner And Invariants

| Object | Owner | Proven invariant |
| --- | --- | --- |
| Time-sorted identity | `src/unilab/algos/torch/distill/formal_identity.py::resolve_time_sorted_formal_output_identity` | Fixed clock + `run_name` deterministically returns one stem and fresh paths without I/O. |
| Spec parsing and freeze metadata | `scripts/deploy/materialize_formal_dagger_gate0.py::load_materialization_spec` | Generated paths enter the ordinary formal spec once; `auto_output_identity` is written to the freeze. |
| One-shot execution | Existing formal supervisor owner | It receives frozen argv only; it never generates a timestamp. |

`run_name` is limited to lower-case letters, digits, `_`, and `-`, beginning
with a letter or digit. A spec may use either `run_name` or explicit
`run_dir`/`artifact_dir`; mixing them fails closed. Explicit manual specs
remain backward compatible.

## Evidence

- S1 owner test: fixed clock resolves
  `20260720-090807_g1_walk_stand_fresh_oom_r2` and its two default roots;
  unsafe names fail.
- S1 connector tests: generated paths are parsed into the formal spec; mixing
  `run_name` and a manual path fails.
- S2 materialization fixture: generated output paths and the exact
  `auto_output_identity` dictionary occur in both result and freeze, with
  `training_executed=false` and both new output directories still absent.
- Fresh local verification, all from
  `/Users/chengyuxuan/ArtiIntComVis/UniLab`:

  ```text
  UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --no-sync pytest \
    tests/algos/test_distill_formal_identity.py \
    tests/scripts/test_materialize_formal_dagger_gate0.py \
    tests/scripts/test_check_formal_dagger_postflight.py \
    tests/scripts/test_check_docs.py -q
  56 passed in 2.22s

  UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --no-sync ruff check ...
  All checks passed!

  UV_CACHE_DIR=/tmp/unilab-uv-cache uv run --no-sync mypy ...
  Success: no issues found in 2 source files

  cd note/architecture/auxiliary/atlas_app && npm run check
  atlas OK runtime_modules=9 method_modules=11 concept_nodes=6
  ```

## Limits And Next Boundary

This evidence proves the local owner and connector contract. It does not prove
server clock, available disk/GPU, Hydra composition for a new fresh-r2 spec,
or training stability. The next boundary is a separate human choice of fresh-r2
workload/resource values and `run_name`, followed by an explicitly authorized
no-training Gate 0 only.
