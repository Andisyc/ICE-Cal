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
| DISTILL-DP-01 | Teacher Policies | StandHeight + Walk, common 99-D input | E120: StandHeight teacher cannot recover lateral@0.702 or satisfy nominal lateral 20-step decay from the exact student-walk switch states |
| DISTILL-DP-02 | Command Intent | inactive preserves target height; active walks at 0.754 m | live 3x3 collection passed; r3 passed 5/9 physical recovery cases |
| DISTILL-DP-03 | Role Data | explicit 99-D role/intent/height/teacher rows | schema, roundtrip, legacy isolation, and per-case metadata passed locally |
| DISTILL-DP-04 | MoE Student | two experts, active->0 and inactive->1 | E120: inactive expert fails three same-state cases that the frozen teacher passes |
| DISTILL-DP-05 | Student-State DAgger | inherited cumulative outer loop | r2 first-invalid-operation is owner-confirmed and repaired; immutable r3 completed |

## Current Step

E120 closes the diagnosis of r3's remaining physical failures without changing
the gate. Exact-switch-state controller substitution separates two owners:
three recovery failures belong to student closed-loop fidelity, while lateral
0.702 and nominal lateral 20-step decay remain unavailable even to the frozen
StandHeight teacher. r3 remains unpromoted and no repair/training is authorized.

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
- E119: unperturbed core captured the first invalid operation, repair commit
  `f9062077` passed local/remote near-risk tests and three exact-r2 save/reload
  repetitions. Immutable r3 completed with 1,310,720 samples and 16,384 updates;
  checkpoint SHA-256 is `f1cbc7d7...909d`. Its unchanged seed-1 gate passed 5/9
  recovery cases but still failed nominal lateral stop decay.
- E120: fixed-seed exact-switch-state substitution produced zero pre-branch
  state difference. The teacher passed forward 0.650/0.702 and lateral 0.754
  where the student terminated, but both controllers failed lateral 0.702 and
  nominal lateral 20-step decay. Artifact SHA-256 is `b09918b2...f71`.

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

- r1 and failed r2 remain immutable evidence; the r2 native defect is closed by
  first-invalid-operation evidence and exact-source regression.
- r3 is unpromoted. E120 separates the red gate into a DP-04 student-fidelity
  boundary and a DP-01 teacher recoverable-domain/transition compatibility
  boundary; neither has an authorized repair.
- `.codex-tmp/` and `.local-build/` remain excluded from source changes.

## Next Step

Stop before another material training run. Preserve r1, failed r2, and completed
r3 as immutable evidence. The policy-quality diagnosis is closed by E120. The
next step is a human method decision: address both student closed-loop fidelity
and the teacher-insufficient lateral recovery boundary before defining a new
immutable training fork.
