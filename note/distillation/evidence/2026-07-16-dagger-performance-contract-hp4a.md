# DAgger Performance Metrics Contract HP-4a Evidence

Date: 2026-07-16

Scope: pure distillation-owned metric identity, stage schema, injected-clock
assembly, validation, derived throughput, atomic JSON persistence, and reload.
No collector/workflow/worker instrumentation, MuJoCo, training, or performance
comparison is included.

## Design And Owner Decision

HP-4a is an engineering evidence layer under the existing `Teacher Policies`,
`Role Data`, and `Student-State DAgger` design points. It does not add a method
block or change `DISTILL-METHOD-v001` / `DISTILL-TRAIN-v002` semantics.

The repository's generic `TraceRecorder` owns Chrome/Perfetto events but does
not validate DAgger checkpoint, teacher, config, request, or weight-version
identity. Therefore `src/unilab/algos/torch/distill/performance.py` owns the
domain schema and run-local artifact. Runtime owners remain unmodified.

## Metric Contract

Each record carries execution mode, outer iteration, scenario, PID, request,
checkpoint path/hash, nullable-by-mode weight version, teacher hashes, config
hash, seed, device, `num_envs`, stage, seconds, row/env-step counts,
success/error, cleanup state, and derived rows/second.

Canonical stages cover cold start, weight sync, env init/reset/step, student
and teacher inference, tensor packing, artifact write, cumulative aggregation,
learner staging/forward/backward/optimizer, checkpoint save, and total elapsed.

Fail-closed rules cover invalid hashes, negative/non-finite durations and
counts, legacy/persistent weight-version mismatch, run identity drift,
same-request checkpoint/PID/version drift, missing required stages, incompatible
duplicate records, invalid error/cleanup combinations, and derived-rate drift.

## Evidence

Test-first red: import failed with
`ModuleNotFoundError: unilab.algos.torch.distill.performance`.

Focused green:

```text
uv run pytest tests/algos/test_distill_performance.py -q
16 passed in 0.04s
```

Impact groups:

- performance + async/persistent/workflow: `39 passed in 4.11s`;
- distill entry/config: `70 passed, 250 deselected in 3.84s`.

Ruff: `All checks passed!`. Atlas:
`runtime_modules=9 method_modules=11 concept_nodes=6`.

## Decision

HP-4a's pure metrics contract is implemented and locally accepted. It is
`implemented-not-integrated`: no runtime stage currently emits this schema.
Control returns to the user before designing connectors or freezing Gate 0B.
No speedup or bottleneck claim is authorized.
