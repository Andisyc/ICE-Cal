# Distillation Task Canvas

## Objective

Implement the active `DISTILL-METHOD-v002`: train one `StandHeight` behavior
teacher for zero-velocity target-height tracking, combine it with one `Walk`
teacher, and distill both into a unified 99-D, two-expert MoE student.

## Current Concept Figure

- Path: `note/architecture/concept/03_g1_multiteacher_distillation_method.data.json`.
- Update owner: current Codex task under human-confirmed v002 semantics.
- Active method contract:
  `note/distillation/contracts/active/method/DISTILL-METHOD-v002.md`.
- Inherited training lifecycle:
  `note/distillation/contracts/active/training/DISTILL-TRAIN-v003.md`.

## Design Point Register

| Design ID | Canonical name | Active meaning | Current gap |
| --- | --- | --- | --- |
| DISTILL-DP-01 | Teacher Policies | StandHeight + Walk, common 99-D input | teacher and role-artifact identities were reread and hashed on SSH; no teacher-quality gap is currently observed |
| DISTILL-DP-02 | Command Intent | inactive preserves target height; active walks at 0.754 m | live 3x3 collection passed; the first retrained child passed 4/9 physical recovery cases |
| DISTILL-DP-03 | Role Data | explicit 99-D role/intent/height/teacher rows | schema, roundtrip, legacy isolation, and per-case metadata passed locally |
| DISTILL-DP-04 | MoE Student | two experts, active->0 and inactive->1 | repaired r1 child exists but failed nominal lateral stop decay and 5/9 non-nominal recovery cases |
| DISTILL-DP-05 | Student-State DAgger | inherited cumulative outer loop | r1 completed; r2 stopped before aggregate completion on a native SIGSEGV boundary |

## Current Step

The repaired grid is now live-confirmed through one immutable SSH fork. E118
records the completed r1 child, exact hashes, nine-case collection metadata,
and unchanged MuJoCo acceptance. The child improved offline expert imitation
but passed only 4/9 non-nominal cases and failed nominal lateral stop decay.
A second immutable fork collected the child's states but exited at
`before_aggregate`; an offline aggregate replay passed in memory, while the
save/reload replay produced exit 139 and became timing-sensitive under GDB.

Current repair plan:
`note/distillation/plans/non_nominal_transition_dagger_repair.md`.

Completed pre-training baseline plan:
`note/distillation/plans/stand_height_walk_two_teacher_implementation.md`.

Acceptance checklist:
`note/distillation/checklists/stand_height_walk_two_teacher.md`.

## Verified Evidence

- Step 1 governance: `DISTILL-METHOD-v002` and all six Concept Figure nodes map
  to the active two-role method; the Atlas contract check passes.
- E114: the retained Step 2 record reports `108 passed, 24 warnings in 19.46s`;
  the Step 3 adapter/connector record reports `8 passed in 6.77s`. E114 also
  preserves their exact transcript turn IDs and evidence limitations.
- E113: Step 4 Ruff PASS and `27 passed in 20.56s`; 99-D owner profiles,
  `walk|stand_height`, `active->0|inactive->1`, `(N, 1) target_height`
  roundtrip, selected-expert isolation, strict checkpoint reload, and legacy
  98-D isolation are deterministic PASS.
- E115: Step 5 focused suite `5 passed in 4.21s`, Ruff PASS, one-env/one-step
  MuJoCo sentinel exit 0, and fixed-height training compose preflight exit 0.
- E116: async owner connector suite `3 passed in 3.83s` and focused Ruff PASS;
  99-D roles, target-height keys, all three scenarios, resident service routing,
  and cleanup are connector-confirmed with synthetic fixtures.
- E117: transition core-path suite `10 passed, 95 deselected in 0.71s`,
  workflow/persistent connector suite `7 passed in 4.64s`, and focused Ruff
  lint/format PASS. Nine command-height cases, active `0.754 m`, post-switch
  `0.650/0.702/0.754 m`, observation index 96, teacher relabeling, per-case
  horizon evidence, legacy fallback, and both connectors are deterministic PASS.
- E118: SSH r1 completed with `1,114,112` cumulative samples and `12,288`
  updates; checkpoint SHA-256 is `13378dd0c7c7478307692b775bd72305fb1a4bfd2d2fffe7e1a96d1ca84844f9`.
  All nine collection cases have post-switch rows through age 968. The fixed
  seed physical gate failed: nominal lateral stop decay failed and non-nominal
  recovery passed 4/9. The r2 aggregate identity is source-valid at 1,310,720
  rows, but formal aggregation and one save/reload replay exposed a native
  SIGSEGV; GDB changed timing and the same replay passed.

## Active Files And Commands

- Contract/figure: v002 contract and `03_g1_multiteacher_distillation_method.data.json`.
- Current plan/checklist: `non_nominal_transition_dagger_repair.md` and
  `stand_height_walk_two_teacher.md`.
- Repair owners: `conf/distill/workflow/g1_stand_height_walk.yaml`,
  `src/unilab/algos/torch/distill/collector.py`, `scripts/train_distill.py`,
  and `src/unilab/algos/torch/distill/g1_persistent_worker.py`.
- Deterministic evidence owners:
  `tests/algos/test_g1_distillation_contract.py`,
  `tests/algos/test_distill_persistent_differential.py`,
  `tests/algos/test_distill_g1_persistent_worker.py`, and
  `tests/scripts/test_stand_height_walk_distill_workflow.py`.
- Async owners: `src/unilab/ipc/async_runner.py`,
  `src/unilab/algos/torch/offpolicy/double_buffer_runner.py`, and
  `src/unilab/algos/torch/distill/persistent_runtime.py`.
- Live route owner:
  `scripts/deploy/check_unilab_g1_height_tracking_live_path.py`.
- Validation rule: always execute Python tooling through `uv run`.

## Unresolved Risks

- The r1 repaired checkpoint is unpromoted because its governed physical gate
  failed; offline MSE improvement is not policy-quality evidence.
- The r2 child-state fork has no checkpoint. Its source list and in-memory
  aggregate pass, but the save/reload boundary is native-symptom-confirmed and
  not owner-confirmed.
- GDB made the same replay pass, which proves timing/layout sensitivity only.
  Do not retry until the native serialization/ownership boundary has a scoped
  repair or a first-invalid-operation capture.
- `.codex-tmp/` and `.local-build/` remain excluded from source changes.

## Next Step

Stop before another material training run. Preserve r1 and failed r2 as
immutable evidence. The next engineering step is the CPU aggregate
save/reload native boundary: reproduce with the exact r2 identity, obtain an
owner-level repair or first-invalid-operation evidence, then resume through a
new immutable fork and rerun the unchanged fixed-seed physical gate.
