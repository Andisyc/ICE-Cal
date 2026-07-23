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
| DISTILL-DP-01 | Teacher Policies | StandHeight + Walk, common 99-D input | task and actor-only adapter implemented; qualified checkpoints absent |
| DISTILL-DP-02 | Command Intent | inactive preserves target height; active walks at 0.754 m | deterministic routing passed; non-nominal live transition unproven |
| DISTILL-DP-03 | Role Data | explicit 99-D role/intent/height/teacher rows | schema, roundtrip, and legacy isolation passed; qualified real artifacts absent |
| DISTILL-DP-04 | MoE Student | two experts, active->0 and inactive->1 | deterministic workflow/update/reload passed; trained checkpoint absent |
| DISTILL-DP-05 | Student-State DAgger | inherited cumulative outer loop | physical acceptance |

## Current Step

Step 5 / 5 is complete at the bounded live-route boundary. E115 confirms one
real `G1StandHeight` MuJoCo environment with a 99-D actor observation, target
height at index 96, zero velocity command, finite support/tilt/reward facts,
and no one-step termination. No checkpoint was read and training did not start.
The post-closure E116 connector check also confirms that StandHeight SAC uses
the repository async/double-buffer runner and that the new two-role workflow
can explicitly opt into the resident persistent runtime. The distillation
default remains `legacy` because stable speedup is not proven.

Implementation plan:
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

## Active Files And Commands

- Contract/figure: v002 contract and `03_g1_multiteacher_distillation_method.data.json`.
- Current plan/checklist: `stand_height_walk_two_teacher_implementation.md` and
  `stand_height_walk_two_teacher.md`.
- Implemented Step 4 owners: `conf/distill/workflow/g1_stand_height_walk.yaml`,
  the two height-aware role profiles, and distill collector/data/workflow/
  trainer/DAgger connectors.
- Deterministic evidence owners:
  `tests/algos/test_stand_height_walk_distillation.py` and
  `tests/scripts/test_stand_height_walk_distill_workflow.py`.
- Async owners: `src/unilab/ipc/async_runner.py`,
  `src/unilab/algos/torch/offpolicy/double_buffer_runner.py`, and
  `src/unilab/algos/torch/distill/persistent_runtime.py`.
- Live route owner:
  `scripts/deploy/check_unilab_g1_height_tracking_live_path.py`.
- Validation rule: always execute Python tooling through `uv run`.

## Unresolved Risks

- No qualified local `G1StandStill` or `G1WalkFlat` teacher checkpoint is
  currently available for conversion and training.
- `F:\download\dagger_iteration_8.pt` is a student artifact, not an accepted
  teacher source.
- Teacher training and final DAgger are material compute boundaries; connector
  tests cannot establish policy quality.
- Repeated reset, long-horizon height tracking, and non-nominal
  Walk-to-StandHeight behavior remain unexecuted.
- Existing user changes in `README_zh.md`,
  `note/g1_agile_height_distill_moe_migration.md`, `pyproject.toml`, and
  `.local-build/` must remain untouched.

## Next Step

The repository is stopped immediately before material training. The
preflighted next command is the fresh fixed-`0.754 m` SAC command recorded in
the implementation plan; it already selects the async/double-buffer off-policy
runner, and the human will launch it over SSH. Warm-starting,
range expansion, final DAgger, promotion, commit, push, and PR remain outside
the completed pre-training closure.
