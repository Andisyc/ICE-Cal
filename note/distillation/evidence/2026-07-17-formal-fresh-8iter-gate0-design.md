# Formal Fresh Eight-Iteration Gate 0 Design

Date: 2026-07-17

## Scope

Freeze the final clean fresh-training workload and extend the existing FT-0
owner/connector/oracle route to support it. No server materialization,
supervisor execution, environment, collection, learner, checkpoint, or training
occurred.

## Frozen Identity

- Spec: `plans/formal_dagger_fresh_8iter_r1.spec.json`.
- Lineage: `fresh_teacher_bootstrap`; no parent run/checkpoint.
- Role inputs: immutable manifest-reviewed walk/stand teacher datasets.
- Legacy adoption: false; existing artifacts must pass strict REUSE.
- Bootstrap updates: `20000`.
- DAgger outer iterations: `8`.
- Samples per scenario per iteration: `65536`.
- Batch size: `512`; configured floor: `128`.
- Effective schedule:
  `[4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768]`.
- DAgger update total: `147456`; bootstrap plus DAgger total: `167456`.
- Aggregate rows after each iteration:
  `[458752, 655360, 851968, 1048576, 1245184, 1441792, 1638400, 1835008]`.
- Transition max env steps: `24576`; collect envs: `64`.
- Execution mode: explicit `persistent_async`; repository default unchanged.
- New run and artifact directories end in
  `g1_walk_stand_formal_fresh_8iter_20260717_r1`.

## Owner And Acceptance

The formal identity owner now supports distinct `fork` and `fresh` lineage.
For fresh mode the Gate 0 connector loads the real walk/stand role datasets,
constructs the initial walk/static scenario counts, then calls the production
offline replay owner for all eight future iterations. The observed schedule and
total must equal the spec or preflight fails.

Oracle v2 now accepts either frozen parent-checkpoint lineage or fresh bootstrap
checkpoint lineage, then validates every iteration checkpoint chain. The
generated supervisor remains unexecuted.

## Verification

- Fresh identity and strict-artifact contract tests: PASS.
- Real owner-to-Hydra fresh compose: PASS, no training.
- Serialized role-dataset workload discriminator: PASS.
- Fresh bootstrap-to-iteration postflight fixture: PASS.
- Focused formal/workflow suite: 46 passed.
- Ruff and mypy: PASS.

## Decision

Local fresh Gate 0 design/integration PASS. Authenticated server no-training
materialization remains pending. Final live training is not authorized by this
step.

