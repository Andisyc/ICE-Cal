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
| DISTILL-DP-01 | Teacher Policies | StandHeight + Walk, common 99-D input | remote teacher identities and role artifacts are recorded; bytes were not reread locally |
| DISTILL-DP-02 | Command Intent | inactive preserves target height; active walks at 0.754 m | deterministic 3x3 transition grid passed; retrained live transition unproven |
| DISTILL-DP-03 | Role Data | explicit 99-D role/intent/height/teacher rows | schema, roundtrip, legacy isolation, and per-case metadata passed locally |
| DISTILL-DP-04 | MoE Student | two experts, active->0 and inactive->1 | round-2 student exists but failed the governed non-nominal transition gate |
| DISTILL-DP-05 | Student-State DAgger | inherited cumulative outer loop | repaired collection distribution has not run on SSH |

## Current Step

The non-nominal transition DAgger repair is complete at the local deterministic
boundary. E117 records the round-2 student failure, the old collector's
single-forward/fixed-height distribution gap, and the new command x recovery-
height grid routed through both legacy and `persistent_async` connectors.
No local checkpoint, MuJoCo process, learner, or training run was used.

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

- The repaired collector has not produced a remote dataset or checkpoint.
- The current round-2 checkpoint failed four of nine non-nominal recovery cases;
  it remains an unpromoted parent artifact, not acceptance evidence.
- Local deterministic fixtures prove routing and persistence contracts only;
  repeated reset, long-horizon recovery, and physical policy quality require a
  new SSH fork and the governed live gate.
- The implementation is still local and uncommitted. The server cannot use the
  new grid until the selected branch is committed, pushed, and pulled.
- `.codex-tmp/` and `.local-build/` remain excluded from source changes.

## Next Step

Stop before material training. After the human approves a scoped commit/push
and pulls it on SSH, fork one new outer DAgger iteration from
`20260724-110852_stand_height_walk_dagger_round2` with the exact command in the
current repair plan. Then run the same nominal and non-nominal acceptance gates
against the new checkpoint. The current round-2 run must remain immutable.
