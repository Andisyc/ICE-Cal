# Workflow Materialization Repair And r8 Refreeze

Date: 2026-07-17

Status: `PASS` for the workflow-owner repair and no-training r8 freeze. Formal
A/B execution remains separately authorized.

## Owner Repair

E59 proved that the real persistent worker received an output path whose parent
did not exist. The repair stays at the workflow iteration owner:

```text
iteration_dir construction
-> iteration_dir.mkdir(parents=True, exist_ok=True)
-> DaggerCollectRequest.output_path
-> persistent collector artifact write
```

The spawned workflow fake was changed to assert that the parent already exists
and to write directly without creating it. Before the source repair, the exact
spawned-runner test failed at `output_path.parent.is_dir()`. After adding the
single workflow-owner materialization line, the same test passed.

No script, persistent worker, dataset owner, method contract, teacher,
checkpoint, workload, or oracle behavior changed.

## Verification

- Focused spawned regression: `1 passed, 17 deselected`.
- Full workflow tests: `18 passed`.
- Persistent runtime/IPC boundary with POSIX shared memory: `40 passed`.
- Full affected distillation/workflow/runtime/config suite: `493 passed`.
- Ruff on the two touched source/test files: `PASS`.
- `git diff --check`: `PASS`.

## r8 Immutable Identity

- Freeze directory:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r8`.
- Two deterministic source generations are byte-identical.
- Bundle SHA-256:
  `ea1d4f7a6acc3a35f9669bbc55c3df681e48100bdbe72e0880def609f5d5b25e`.
- Source manifest SHA-256:
  `5ba8ce6c5085a2c81de7e25f3dd3bb56b57c69f7aee3d2e6d5faa4429bf3d637`.
- Source file count: `1252`.
- Frozen cwd: `/private/tmp/unilab-hp4b-ea1d4f7a`.
- Identity SHA-256:
  `0dc04b35d7a3ad04b3821372f5f11d30b6eb5d8cabbf780c4798067428e9240e`.
- Oracle v2 SHA-256:
  `9e62b678eb02d792c587b2a46ecc7fae1e000b9376d5bfbc229683170fedb631`.
- Pre-execution oracle contract SHA-256:
  `ad483f800e37ba52900a051e4df831ea08471d7cb58dbd0c1f3bbb5b95d65b6f`.
- Formal output root:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_ab_20260717_gate0b_r8`,
  absent.
- State: `FROZEN_NOT_EXECUTED`, `execution_authorized=false`.

## Frozen No-Training Preflight

Artifact:
`/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r8/frozen_preflight.json`,
SHA-256 `ce037501a10a180f1eedffe96275b5b9dc9a0e8532c721dba345be75e2334dd0`.

- All 1252 frozen-source and external asset hashes match.
- Build produces wheel and sdist.
- Nested provider snapshot/import/entrypoint help pass with 171 packages.
- MuJoCo scene reports `nq/nv/nu=36/35/29`.
- Shared compose hash remains `15417bfa...69ca`; legacy and persistent differ
  only by the frozen execution mode and run directory.
- Generic teacher/student remain 99/99; workflow teacher/student remain 98/98;
  both real checkpoint contracts pass at 98-D input and 29-D action.
- Frozen affected suite reports `493 passed`; Ruff passes.
- `training_started=false` and the r8 formal output root remains absent.

## Decision

The E59 owner defect is repaired and r8 is ready for a new formal A/B sequence.
r7 partial artifacts are retained only as failure evidence and are not resumed.
Because source identity changed, a future execution must begin from r8 order 1
and use the frozen oracle v2 after every successful run. This repair step stops
before any training or A/B execution.
