# HP-7c3 Gate 0 Amendment And Compose PASS

Date: 2026-07-17

Scope: no-training Gate 0 source/artifact/config identity review. No workflow
run, environment construction, collection, learner update, checkpoint, or
promotion action was executed.

## Evidence

- Server HEAD: `4fd2f67c08bb5372221ee1347561145b27238a75`.
- Frozen runtime-owner hash mismatch list: empty. The server HEAD differs from
  the earlier planning HEAD, but the inspected runtime owner bytes do not.
- Parent role datasets are manifest-owned artifacts:
  - walk: `/ssd1/cyx/UniLab/model/teacher/walk_flat_teacher_policy.pt`, SHA-256
    `efa0bec38f43b2ef3e811e1d35fc1f54a40d0d7377aafaa47e113b74aa5be027`;
  - stand: `/ssd1/cyx/UniLab/model/teacher/stand_teacher_policy.pt`, SHA-256
    `f0e37612a74a355e429518e1241cc4c991111deb5e6483bb592b5732dc085b59`.
- Teacher checkpoint SHA-256 values remain
  `db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`
  for walk and
  `91e18d3d1f469b2bead350cd41b33494c39c8ec8d26f2daf802e0273afa2c6da`
  for stand.
- Parent metrics are audit-only for fork materialization. The parent manifest
  records SHA-256
  `6ad722ec2b305de12408d9cffdf464394989c167295a8d1275b18d4f38ba1690`,
  while the observed file is
  `cef25efe6af5752528f1d9d61076de450a310d2589c18ec093699bb3d5959401`.
  `fork_workflow_run()` reads the parent manifest, latest aggregate dataset,
  latest checkpoint, bootstrap sources, role artifacts, and scenario specs; it
  does not read or import the parent metrics artifact. Record this mismatch as
  non-blocking audit evidence, not as a training-input match.
- The human accepted the exact production-derived replay budget:
  `required_updates=12320` and `effective_updates=12320`. The configured Hydra
  floor remains `512`; `auto_expand_replay_budget=True` owns the expansion.
- Formal owner-CLI compose exited `0`, emitted `6795` bytes, and produced
  SHA-256
  `741676aca03cbed11f9ad6e37105216b3acb545b35ebc86690202b2c0798798d`.
  Stderr is empty. Pulled raw artifacts are repository-root files
  `hp7c3_gate0_compose_r2.yaml` and `hp7c3_gate0_compose_r2.stderr`.
- Resolved config confirms seed `0`, device `cuda:0`, workflow enabled, fork
  mode, `persistent_async`, 16 collector envs, 512 samples per role, one outer
  iteration, batch size 512, configured update floor 512, and teacher/student
  observation dimensions 98/98.

## Decision

The formal Hydra compose boundary passes. E96 remains historical evidence for
the failed non-interactive SSH attempt and is no longer the current blocker.
Gate 0 is still PARTIAL until a server-side materializer creates and hashes the
freeze JSON and oracle, runs the no-training oracle preflight, and freshly
confirms that every Gate 1 output path is absent. Gate 1 remains closed.

The repository materializer is
`scripts/deploy/materialize_hp7c3_gate0.py`. Local `py_compile`, Ruff, and
`--help` checks pass; its server preflight remains unconfirmed until execution
in the authenticated server workspace.
