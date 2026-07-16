# Playback Reset And Transition DAgger Repair Plan

Status: approved problem boundary; Bug A engineering-ready; Bug B mechanism
conditional on teacher-authority evidence.

Evidence source:
`../evidence/2026-07-16-reset-and-transition-repair-preflight.md`.

## Design Alignment

| Design point | Current contract | Gap | Planned treatment |
| --- | --- | --- | --- |
| Command Intent (`DT-M-02`) | zero command means standing | reset and cached obs can disagree with visible command | atomic reset/command/obs contract |
| MoE Student (`DT-M-04`) | inactive -> expert 1 | route can use new command while expert sees old command | route and expert consume one synchronized command state |
| Student-State DAgger (`DT-M-05`) | teacher labels student-visited role states | no cross-role recovery states | conditional `walk_to_stop` scenario collection |
| Robot Execution (`DT-X-01`) | stand, walk, and stop recovery | reset and stop gates absent | repeated-reset and transition live acceptance |

No new top-level Concept Figure block is needed. Transition recovery is a
scenario inside Student-State DAgger. If the teacher-authority gate passes, the
figure feedback label may be refined to mention role and transition rollouts,
while preserving `DT-M-05` and `DISTILL-DP-05`.

## Semantic Source Of Truth

| Object | Active owner | Active consumer | Legacy defect | Required proof |
| --- | --- | --- | --- | --- |
| playback reset command | playback initialization + G1 reset provider | reset gait/qvel plan | sampled walking command overwritten after reset | reset command, gait flag, base qvel agree |
| external command update | G1 env command API + playback session refresh | router and student observation | env info and cached obs can differ for one action | same command seen by route and expert |
| transition scenario | `collector.py` transition owner | workflow dispatch | collector/schema implemented; workflow absent | exact walk/switch/recovery schedule |
| recovery target | standing teacher | stand expert behavior loss | authority unmeasured | teacher differential before training |
| transition identity | distillation dataset schema | workflow sampler/diagnostics | schema/collector identity now explicit | scenario label and transition age roundtrip |

## Step Map

### Step RT-1 / 9: Deterministic Reset And Command-Sync Contract

- Objective: encode the two integration invariants before changing behavior.
- Scope: fake playback lifecycle, G1 reset-plan fixture, command-info versus
  policy-observation fixture.
- Non-scope: policy quality and transition training.
- Owner files: playback tests and G1 reset tests.
- Core parameter path: reset distribution -> sampled command -> gait mask ->
  base qvel; key command -> env info -> env obs -> session obs -> router/expert.
- Evidence: S1/S2 regression tests fail on current ordering/skew.
- Stop condition: tests reproduce non-standing reset and route/obs mismatch.

### Step RT-2 / 9: Playback Reset And Atomic Command Fix

- Objective: make keyboard playback start from and remain internally consistent
  with an explicit command.
- Scope: playback-only standing reset distribution, a G1-owned external command
  setter that refreshes state observation, and session observation refresh.
- Non-scope: changing training reset distributions or adding action smoothing.
- Owner files: `joystick.py`, `interactive_playback.py`, `play_interactive.py`.
- Evidence: RT-1 regressions pass; old OFF/non-keyboard behavior is unchanged.
- Stop condition: reset zero command implies gait-disabled, zero base qvel, and
  identical command values in info, actor obs, route diagnostics, and expert input.

### Step RT-3 / 9: Repeated-Reset Live Sentinel

- Objective: prove Bug A on real MuJoCo before touching training.
- Scope: 32 reset-only repetitions and the first policy action boundary.
- Non-scope: long standing or walking quality.
- Probe fields: reset index, command, actor-observation command, gait flag,
  base linear/angular qvel, expected/selected expert, checkpoint hash.
- Evidence: E18/S4 live sentinel; all 32 resets satisfy the standing contract.
- Stop condition: any non-zero reset command/qvel or command mismatch blocks RT-4.

### Step RT-4 / 9: Standing-Teacher Recovery Authority Differential

- Objective: determine whether transition DAgger has a valid teacher oracle.
- Scope: exact single-env snapshot replay with static SS and WT/WS branches,
  restore one exact post-walk snapshot into WT and WS with a zero-command
  switch, short horizons, and no optimization.
- Non-scope: dataset persistence and student updates.
- Probe fields: source controller, pre-switch command, switch step, transition
  age, base height, tilt, linear/angular velocity, termination, teacher/student
  action MSE.
- Evidence: E19 checkpoint preflight and E21 final exact-snapshot differential run.
- Stop condition: WS standing-teacher recovery passes to authorize RT-5;
  otherwise Bug B remains blocked on a recovery-capable teacher/curriculum
  decision.

### Step RT-5 / 9: Versioned Transition Training Contract

- Objective: activate the training semantics selected by RT-4.
- Scope: activate the approved `DISTILL-TRAIN-v002` contract and synchronize
  the registry, history, checklist, evidence, and current Architecture view.
- Non-scope: transition schema and collector implementation.
- Owner files: `contracts/active/training/DISTILL-TRAIN-v002.md`, the archived
  v001 contract, and the transition decision record.
- Contract if authorized: walk expert executes pre-switch; zero command is
  applied atomically; stand expert executes post-switch student states;
  standing teacher labels only post-switch recovery rows.
- Forbidden: treating transition as a third expert role; labeling active rows
  with standing teacher; adding blending without differential evidence.
- Evidence: E22 proposal plus E23 contract activation.
- Stop condition: one active v002 registry entry and no stale v001 active copy.

### Step RT-6a / 9: Transition Dataset Schema Owner

- Objective: represent recovery evidence without losing semantics at the data
  boundary.
- Scope: row-level `scenario_label=walk_to_stop`, `transition_age`, source
  command, post-switch zero command, cached standing-teacher action, roundtrip.
- Non-scope: live rollout, workflow, and optimization.
- Owner files: `distill/data.py` and `distill/trainer.py` batch transport;
  normal collection remains unchanged by default.
- Evidence: E24 schema roundtrip and malformed-row contract tests.
- Stop condition: saved/reloaded rows preserve scenario and switch chronology.

### Step RT-6b / 9: Transition Collector Owner

- Objective: collect the rows defined by RT-6a from student-state rollouts.
- Scope: atomic command switch, walking/standing teacher selection, done/reset
  handling, bounded sample counts, and transition metadata.
- Non-scope: workflow integration and optimizer changes.
- Owner files: `distill/collector.py` plus a focused collector probe/test;
  normal collector remains unchanged by default.
- Evidence: E25 tiny golden lifecycle with active pre-switch and inactive
  recovery rows; shape, finite, command, role, teacher, and reset assertions.
- Stop condition: a saved transition dataset passes RT-6a validation and its
  runtime trace records the switch exactly once per episode.

### Step RT-7 / 9: Workflow Integration And Hierarchical Balancing

- Objective: add transition scenarios to every DAgger outer iteration without
  pretending they are a new role.
- Scope: workflow scenario specs, manifest identity, cumulative aggregation,
  resume/fork, and configurable batch quotas.
- Recommended initial quota: 50% walking expert; 50% standing expert split into
  25% static standing and 25% walk-to-stop recovery.
- Non-scope: hard-coded final research ratio or height-control integration.
- Evidence: iteration `k+1` uses checkpoint `k`; static/recovery/walk counts and
  selected experts match quotas; scenario artifact hashes are persisted;
  resume/fork rejects mismatched scenario specs; legacy workflow OFF path is
  unchanged.
- Result: E26 passes the bounded CPU/connectivity gate and the formal Hydra
  profile composes all three scenarios with `0.50/0.25/0.25` quotas. No real
  G1 transition dataset or student physical result is implied.
- Stop condition: RT-7 is complete at S2. Continue only to RT-8 for the bounded
  real-artifact workflow and physical acceptance grid.

### Step RT-8 / 9: Bounded Training And Physical Acceptance

- Objective: produce a candidate and decide promotion using repeated physics,
  not file size or offline MSE.
- Scope: bounded full workflow run followed by reset, stand-to-walk, low-speed,
  and walk-to-stop grids.
- Minimum gate: 32 starts; at least 20 transitions covering forward, lateral,
  and yaw commands; no termination; base height/tilt within declared task
  limits; velocity decays after stop; checkpoint/teacher/manifest hashes logged.
- Non-scope: height teacher or final universal-policy quality.
- Result: E27 completes the bounded real-artifact workflow and persists the
  candidate lineage. E28 executes the physical grid but fails promotion with
  repeated termination and no stop-speed decay.
- Stop condition: not met. Keep the candidate unpromoted and return to the
  student transition training/data owner before another physical rerun.

### Step RT-9a: Student-Transition Failure Mechanism Audit

- Objective: identify the first failed transition dataflow boundary before
  changing collection or optimization behavior.
- Scope: audit transition age coverage, done/failure-state coverage, manifest
  update exposure, cached target behavior, hard command routing, and a minimal
  three-command MuJoCo sentinel.
- Non-scope: model-structure changes, stand/walk role recollection, unbounded
  training, or checkpoint promotion.
- Owner files/artifacts: `collector.py`, `data.py`, `offline.py`,
  `trainer.py`, the RT-8 manifest, transition dataset, candidate checkpoint,
  and the bounded live probe.
- Evidence: E29 and
  `logs/distill_workflow/rt8_bounded_20260716_retry4/rt9a_transition_audit.txt`.
- Result: schema, dimensions, normalizer state, and playback hard routing are
  not the first failure. The transition artifact ends at post-switch age 7,
  contains no done/failure rows, and receives only four optimizer updates with
  a 25% transition quota.
- Stop condition: met. RT-9b is the next bounded owner step; do not rerun the
  physical grid before repairing coverage and training exposure.

### Step RT-9b: Transition Coverage And Training Exposure Repair

- Objective: prevent a transition artifact from passing schema validation while
  lacking the post-switch states and optimizer exposure required by the live
  acceptance horizon.
- Scope: collector horizon contract, transition metadata, balanced replay
  budget calculation, workflow/profile wiring, and focused regression tests.
- Non-scope: role artifact recollection, model structure, normalizer behavior,
  bounded retrain, or physical acceptance.
- Owner files: `collector.py` owns horizon/state coverage;
  `offline.py` owns expected replay-budget validation;
  `scripts/train_distill.py` and Hydra dispatch the two contracts.
- Evidence: E30; affected distill suite `275 passed, 8 skipped`, targeted
  tests `5 passed`, Ruff and py_compile pass.
- Result: the profile now requires 20 post-switch steps and 8 expected
  transition replay passes. The former RT-8 budget computes as
  `required_updates=64` and fails closed at `max_updates=4`.
- Stop condition: met. Existing stand/walk artifacts remain reusable; RT-9c
  is the next bounded retrain and live-acceptance step.

### Step RT-9c: Reused-Role Bounded Retrain And Physical Gate

- Objective: exercise the strengthened transition contract in the formal
  workflow without recollecting the reusable stand/walk role artifacts, then
  rerun the identical MuJoCo acceptance grid.
- Scope: one fresh workflow run, role manifest reuse, student-state role and
  walk-to-stop collection, transition horizon enforcement, balanced replay,
  checkpoint lineage, and the 32-reset forward/lateral/yaw live gate.
- Non-scope: height teacher, role artifact recollection, unbounded training,
  checkpoint promotion, or model-structure changes.
- Owner files/artifacts: `workflow.py` and `train_distill.py` for dispatch and
  lineage; `collector.py` and `offline.py` for the strengthened contract;
  `logs/distill_workflow/rt9c_bounded_20260716_run2/` for the run and live log.
- Expected evidence: reused role decisions, 640-row aggregate, transition
  ages reaching at least 20 post-switch steps, 128 DAgger updates, and the
  same physical gate result.
- Result: workflow and data/training-exposure requirements pass. The exact
  candidate reaches MuJoCo but fails the physical gate with 26 terminations
  across 32 resets and `stop_speed_decay_pass=false`. The candidate remains
  unpromoted.
- Stop condition: partially met. The RT-9b owner contract is integrated and
  exercised, but policy-quality acceptance is not met. The next bounded step
  must isolate target quality, student rollout distribution, and router/role
  behavior before another retrain.

### Step RT-9d: Student-Transition Policy-Quality Isolation

- Objective: identify the first failed policy-quality boundary after RT-9c,
  separating teacher-target mismatch, student-state distribution drift, and
  raw-router versus deployment-selected-expert behavior.
- Scope: one offline structured probe over the RT-9c bootstrap aggregate,
  DAgger aggregate, and `walk_to_stop` artifact; force expert 0/1 separately;
  report target MSE by role/intent/transition age; report raw router choice and
  observation distribution summaries.
- Non-scope: training-loss changes, collector changes, role artifact
  recollection, checkpoint promotion, or MuJoCo reruns.
- Owner files: `MoEStudentPolicy.forward` and distillation dataset metadata are
  read-only owners in this step; the new deploy probe owns only measurement and
  formatting.
- Expected evidence: one structured log that can distinguish target quality,
  rollout distribution, and router/role consistency, or explicitly records the
  boundary as unconfirmed.
- Stop condition: name one first failed boundary with runtime facts before any
  new retrain; if no boundary is isolated, stop and return to a design review.

- Result: E32 identifies the first failed boundary. The `walk_to_stop`
  transition workflow passes the full soft student into collection, unlike
  ordinary role DAgger's fixed expert rollout and unlike deployment's
  command-selected expert. The next step is RT-9e command-conditioned rollout
  repair; no retrain was started in RT-9d.
- Stop condition: met. Do not alter target teachers or role artifacts before
  repairing the transition rollout authority.

### Step RT-9e: Command-Conditioned Transition Rollout Repair

- Objective: make the `walk_to_stop` student-state rollout use the same
  command-intent expert authority as deployment and the active Student-State
  DAgger contract.
- Scope: add an explicit active/inactive rollout-policy map at the transition
  collector boundary; select active expert rows before the atomic switch and
  inactive expert rows after it; persist the selected expert mapping in
  collection metadata; add a tiny semantic collector fixture and workflow
  connector regression.
- Non-scope: teacher checkpoints, role artifacts, transition horizon/replay
  quotas, raw router loss, model architecture, or physical retraining.
- Owner files: `collector.py` owns row-wise intent selection; `dagger.py` or
  the existing MoE owner supplies fixed expert modules; `scripts/train_distill.py`
  only resolves the checkpoint's declared command-intent expert targets and
  dispatches the map.
- Core parameter path: `post_switch` state -> active/inactive rollout module
  -> action rows -> env.step -> saved transition metadata.
- Expected evidence: a toy transition collector proves pre-switch actions are
  expert 0 and post-switch actions are expert 1; workflow metadata records the
  mapping and `rollout_policy=command_intent_experts`; legacy role DAgger tests
  remain unchanged.
- Stop condition: implementation and connector tests pass with no unbound full
  student rollout remaining in the transition workflow. Do not retrain until
  this condition is met.

- Result: E33 passes. The collector now supports an explicit intent-to-expert
  rollout map, the workflow resolves it from checkpoint runtime config, and
  the semantic fixture proves active rows use the active policy before the
  switch while inactive rows use the inactive policy afterward.
- Stop condition: met. The next bounded step is RT-10 retrain and physical
  acceptance; do not reuse the RT-9c candidate as a promoted artifact.

## Why The Steps Stay Separate

The work crosses four independent evidence boundaries: deterministic playback,
real reset physics, teacher-oracle validity, and training/data/workflow changes.
Combining them would make a failed live result unable to distinguish lifecycle,
teacher, dataset, sampler, or policy causes.

## Current Next Action

RT-1 through RT-8 implementation and bounded workflow gates are complete, and
RT-9a isolated the first failed transition data boundary and RT-9b repaired its
collection/training guards. E23-E27 record v002 activation, schema/collector
validation, scenario dispatch, manifest lineage, quota balancing, and a real
candidate. E28 is the physical failure evidence, E29 is the dataflow audit,
E30 is the owner repair evidence, E31 shows that the repaired owner is
exercised but does not yet produce a physically accepted candidate, and E32
identifies the transition rollout authority mismatch, and E33 records its
implementation repair. The next action is RT-10 bounded retrain and physical
acceptance; do not start an unbounded run or promote the RT-9c candidate.
