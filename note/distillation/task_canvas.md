# Distillation Task Canvas

Objective: replace the seven-command manual distillation procedure with one
resumable, manifest-backed training entry while preserving the active
multi-teacher method contract.

Concept Figure: `note/architecture/concept/03_g1_multiteacher_distillation_method.data.json`

Current step: RT-9e command-conditioned transition rollout repair is complete
after RT-9c's bounded workflow and blocked physical gate. The workflow reused
both role artifacts and persisted a complete new lineage; RT-9d identified the
authority mismatch and RT-9e repaired it before retraining.

Completed steps:

- classified active contracts, proposals, checklists, evidence, logs, and history;
- created the active Concept Figure and matching contract register;
- created Method-to-Code and Runtime atlases;
- added repository-local source navigation and static contract checks;
- completed rendered browser source-link and server/editor acceptance;
- redrew the Concept Figure as the selected Causal Spine and added geometry regression checks;
- merged the redundant `Single Policy` block into `MoE Student` without changing runtime semantics;
- replaced the UniLab flow-tree runtime map with 01 runtime-ordered reading cards;
- reduced the Atlas entry to 01 Runtime, 02 Method-to-Code, and 03 Concept Figure;
- promoted `note/distillation/README.md` as the default control-room entry.
- confirmed the playback reset/command ordering defect behind restart-sensitive
  initial standing;
- confirmed the missing transition-conditioned DAgger distribution behind the
  walk-to-stop recovery gap;
- reduced the default document path to Concept Figure, active contract, and
  this current canvas;
- archived completed or absorbed plans outside the active `plans/` surface.
- distinguished outer DAgger iterations from inner optimizer updates and
  confirmed the existing `training.online_dagger=true` iterative code route;
- classified Bug 3 as a public-workflow under-iteration defect: the iterative
  owner exists, but the default route still exposes one-shot diagnostic
  collect/merge/update branches;
- classified the missing DAgger loop annotation in the Concept Figure as a
  `figure-mismatch` without adding an independent Finetune method block.
- added the proposed role-aware artifact preflight so a new height role can
  collect only height data while compatible stand/walk datasets are reused.
- completed RT-1 deterministic probes for active reset distribution and
  command-observation skew; no production behavior changed.
- completed RT-2 owner implementation and focused tests;
- completed RT-3 task-identity repair and 32/32 repeated-reset MuJoCo sentinel.
- completed RT-4 teacher checkpoint preflight and exact WT/WS/SS live differential.
- activated `DISTILL-TRAIN-v002` and archived `DISTILL-TRAIN-v001`;
- implemented transition schema fields, batch transport, persistence, and
  fail-closed multi-source merge with focused contract tests.
- implemented the opt-in transition collector with atomic command switching,
  teacher-role selection, age chronology, and done-row reset handling.
- completed RT-9a audit: transition schema and hard routing are valid, but the
  collected student-state transition horizon ends at age 7, has no done/failure
  rows, and receives only a small bounded training exposure.
- completed RT-9b owner repair: transition collection enforces the configured
  post-switch horizon, and balanced offline training rejects insufficient
  transition replay exposure.

Active files:

- `note/distillation/README.md`
- `note/distillation/contracts/active/method/DISTILL-METHOD-v001.md`
- `note/distillation/contracts/active/training/DISTILL-TRAIN-v002.md`
- `src/unilab/algos/torch/distill/data.py`
- `src/unilab/algos/torch/distill/collector.py`
- `src/unilab/algos/torch/distill/workflow.py`
- `src/unilab/algos/torch/distill/offline.py`
- `src/unilab/algos/torch/distill/trainer.py`
- `scripts/train_distill.py`
- `scripts/deploy/check_unilab_g1_distill_student_transition_live.py`
- `note/distillation/checklists/current.md`
- `note/distillation/evidence/current.md`
- `note/distillation/plans/transition_training_contract_proposal.md`
- `note/distillation/evidence/2026-07-15-playback-reset-and-stop-transition-root-causes.md`
- `note/distillation/evidence/2026-07-16-reset-and-transition-repair-preflight.md`
- `tests/envs/locomotion/g1/test_gait_constraint.py`
- `tests/visualization/test_interactive_playback.py`
- `src/unilab/base/np_env.py`
- `src/unilab/visualization/interactive_playback.py`
- `scripts/play_interactive.py`
- `scripts/deploy/check_unilab_g1_distill_teacher_recovery_differential.py`
- `note/distillation/plans/single_entry_training_workflow_proposal.md`
- `note/architecture/`

Active commands:

- `npm run check` in `note/architecture/auxiliary/atlas_app/`;
- local atlas server at `node auxiliary/atlas_app/serve_architecture.mjs`.

Verified evidence:

- E1 current owner map is code-confirmed by CodeGraph.
- E2 three local candidate checkpoints are artifact-confirmed.
- E3 restart/stop instability is human-observed live evidence.
- E4 one zero-command 100-step differential is runtime-confirmed but narrow.
- E5 current generic sentinel is insufficient for physical acceptance.
- E6 atlas and document contracts pass static consistency checks.
- E7 rendered source navigation reaches the exact repository file and line.
- E8 Causal Spine geometry and browser readability are confirmed.
- E9 playback reset occurs before the commander establishes zero command; the
  current log proves non-zero initial base motion under a visible zero command.
- E10 the generic online DAgger owner repeatedly executes rollout, cumulative
  aggregation, and optimization; the manual collect/offline route is only one
  outer iteration unless explicitly repeated.
- E14 the single-entry workflow also owns multi-round lineage/resume/fork, but
  remains opt-in while `training.workflow.enabled=false` is the default.
- E15 RT-1 probes reproduce an active walk reset (`gait_enabled=1.0`,
  `base_qvel_norm=0.653136`) and active routing with an inactive cached policy
  observation; both targeted probe commands pass.
- E16 RT-2 implementation and focused contract tests pass: `258 passed`, Ruff
  passes, and the command sync path refreshes env and session observations.
- E18 RT-3 MuJoCo repeated-reset sentinel passes for 32/32 resets.
- E19 both real teacher checkpoints are 98-D/29-D and have no normalizer.
- E21 RT-4 final exact-snapshot WT/WS/SS differential passes all 14 checks.
- E22 RT-5 v002 proposal records the accepted transition decisions.
- E23 v002 is the only active training contract; v001 is archived.
- E24 transition schema roundtrip and malformed-row tests pass (`70 passed`).
- E25 transition collector fake-vectorized probe passes; the default collector
  remains unchanged.
- E26 RT-7 scenario workflow tests pass: role/static/walk_to_stop dispatch,
  cumulative aggregation, weighted scenario quotas, manifest artifact hashes,
  resume mismatch fail-closed, and legacy workflow OFF-path compatibility.
- E26 formal Hydra compose exposes the three configured scenarios with quotas
  `walk_flat=0.50`, `static_stand=0.25`, and `walk_to_stop=0.25`.
- E27 bounded real workflow completes one DAgger iteration: bootstrap `256`
  samples, cumulative `448` samples, real transition dataset, checkpoint, and
  run manifest are persisted.
- E28 MuJoCo live grid reaches the candidate checkpoint for 32 resets and
  forward/lateral/yaw transitions; the gate fails with `done_count=19` and
  `stop_speed_decay_pass=false`.
- E29 transition audit records `transition_ages=-1,0..7`,
  `done_seen_samples=0`, manifest `updates=4`, and a three-command live probe
  with forward stop-speed failure plus lateral/yaw terminations.
- E30 focused implementation gate passes: `275 passed, 8 skipped`, targeted
  horizon/replay tests pass, and the actual RT-8 dataset requires 64 transition
  replay updates under the new contract.
- E31 RT-9c reuses the existing stand/walk artifacts, completes a bounded
  one-iteration workflow with 640 aggregate rows and 128 updates, and proves
  the strengthened transition horizon (`post_switch_rows=96`, max age 23).
  The same physical gate still fails with `done_count=26` and no stop-speed
  decay, so the candidate remains unpromoted.
- E32 shows that ordinary role DAgger uses fixed experts but `walk_to_stop`
  passes the soft full student into transition rollout. Candidate raw routing
  selects expert 0 for all rows; forced expert 1 is materially closer to
  standing targets on post-switch rows.
- E33 fixes the transition collector contract with explicit active/inactive
  rollout expert modules. The semantic fixture and full affected suite pass;
  no new transition artifact or live quality claim exists yet.

Unresolved risks:
- real transition collection and the strengthened post-switch horizon are
  proven by E27/E31, but the RT-9c student still fails the walk-to-stop gate;
- the candidate's internal raw router still leaves expert 1 unused on the
  aggregate, although playback hard routing selects the intended expert;
- generic height-observation schema migration remains unimplemented and fails
  closed;
- Concept Figure now exposes the DAgger outer-loop feedback semantics;
- missing promoted-checkpoint owner;
- absent height teacher.
- promoted-checkpoint ownership remains unresolved.

Next action: RT-10 must run one bounded retrain using the reusable role
artifacts, inspect the new transition metadata, and rerun the same MuJoCo gate.
Do not promote the RT-9c candidate or launch an unbounded training job.
