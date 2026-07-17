# Workflow Teacher Config Repair And r7 Refreeze

Date: 2026-07-17

Status: `PASS`. HP-4b remains separately human-gated.

## Scope and owner

Repair E55 at the Hydra workflow owner. The `g1_walk_stand` workflow now
declares the shared 98-D teacher architecture required by both role tasks and
both frozen checkpoints. No Python fallback, task YAML edit, checkpoint-guard
weakening, r6 partial reuse, simulator, collection, or learner update is part
of this step.

## TDD evidence

The existing workflow-profile test now asserts teacher/student 98/98. Before
the owner repair it failed exactly with `AssertionError: assert 99 == 98`.
The only config repair is `teacher.obs_dim: 98` in
`conf/distill/workflow/g1_walk_stand.yaml`, beside the existing 98-D student
owner. The generic distillation owner remains teacher/student 99/99.

Focused workflow/generic isolation passes. The affected config + entrypoint
suite reports `313 passed, 8 skipped`; Ruff passes.

## Real checkpoint contract

Artifact:
`/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/teacher_contract_preflight.json`.

SHA-256:
`f7ba3c5adb714374210310071cc523a78a9231e4d5554b76ba5ad436e2fa6572`.

- Composed workflow teacher/student: 98/98.
- Walk role task spec and checkpoint actor input: 98/98; production guard pass.
- Stand role task spec and checkpoint actor input: 98/98; production guard pass.
- Both action specs remain 29-D; checkpoint hashes match the frozen workload.

## r7 frozen identity

- Bundle SHA-256, identical across two generations:
  `3ae830b279d1ba6fe321107e09192e5f236d62dc6e85c057e0679f11f0344819`.
- Source manifest SHA-256:
  `0fbfd9a89de93291a0345c364d87debdc0e1654ff0603dd9d8ad47735ec55b70`.
- Source file count: 1248.
- Identity:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/gate0b_identity_manifest.json`.
- Identity SHA-256:
  `9b180b464433e0f29e59060c9245e9fbcd1879d988eeab802cee67be22f59718`.
- Frozen cwd: `/private/tmp/unilab-hp4b-3ae830b2`.
- Formal output root:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_ab_20260717_gate0b_r7`.
- State: `FROZEN_NOT_EXECUTED`, `execution_authorized=false`.

Compose hashes are shared `15417bfa...69ca`, legacy `53b0329d...8684`, and
persistent `b6b571fc...a5b5d`; only mode and run dir differ by route.

## Frozen-cwd preflight

- Build produces sdist and wheel.
- All 1248 source hashes and seven asset hashes match.
- G1 XML loads with `nq/nv/nu=36/35/29`.
- Frozen compose and real two-role teacher contract pass.
- Exact nested provider snapshot/import/entrypoint preflight passes; artifact
  SHA-256 `aef318c914e16f43084c9ec3c5696f9639d452f9b7019cc826e4bf2832795867`.
- Frozen affected suite: `313 passed, 8 skipped`.
- Formal r7 output root remains absent; training did not start.

One helper launch initially selected the extracted incomplete local venv and
stopped during dependency acquisition before the helper ran. Launching the
same helper through the frozen r7 provider/cache/no-sync identity passed. This
did not alter source, identity, workload, or output state.

## Decision

The config-owner repair and r7 refreeze pass. E55 is fixed at the Hydra owner
while generic 99-D behavior and the checkpoint guard remain intact. Control
returns before HP-4b; r6 partial artifacts are neither resumed nor reused.
