contract_id: DISTILL-TRAIN-v002
status: superseded decision record
effective_date: 2026-07-16
updated_date: 2026-07-16
supersedes: DISTILL-TRAIN-v001 on activation only
active_contract: ../contracts/active/training/DISTILL-TRAIN-v003.md
method_contract: DISTILL-METHOD-v001
concept_figure: note/architecture/concept/03_g1_multiteacher_distillation_method.data.json

# Transition-Conditioned Distillation Training Contract Proposal

This file is retained as the decision record for the accepted contract. The
active semantic source is the linked v002 contract; implementation status is
tracked in the current checklist and evidence files.

## Design Alignment

| design_id | concept block | role in this proposal |
| --- | --- | --- |
| DISTILL-DP-02 | DT-M-02 Command Intent | zero command is standing intent; any non-zero planar or yaw velocity is walking intent |
| DISTILL-DP-04 | DT-M-04 MoE Student | active command routes to walk expert 0; inactive command routes to stand expert 1 |
| DISTILL-DP-05 | DT-M-05 Student-State DAgger | walk-to-stop is a labelled student-state scenario inside DAgger, not a third expert |
| DISTILL-DP-01 | DT-M-01 Teacher Policies | walking teacher labels pre-switch walking states; standing teacher labels post-switch recovery states |
| DT-X-01 | Robot Execution | zero-command switch and physical recovery are evaluated as a separate acceptance boundary |

No new top-level Concept Figure block is proposed. The transition scenario is
owned by Student-State DAgger and feeds Role Data. The existing method figure
must remain unchanged unless the human later changes the method meaning.

## Problem Boundary

The current role-conditioned dataset contains static standing and walking
states, but it does not contain student states visited immediately after a
walking-to-zero command switch. Hard routing can select the stand expert while
the stand expert has never seen those post-walk states. This proposal closes
that distribution gap without changing the two-role method.

E21 supports the recovery-oracle prerequisite for one forward scenario:
walking teacher -> zero command -> standing teacher, with 80 pre-switch and
80 post-switch MuJoCo steps at walk-vx=0.4. Lateral, yaw, and student-policy
recovery remain unconfirmed.

## Proposed Scenario Semantics

### 1. Pre-switch walking phase

- The student policy generates the rollout state.
- Any non-zero command is labelled active and owned by walk_flat.
- The walking teacher is queried for the action target.
- The walking expert remains the selected behavior-loss expert.

### 2. Atomic switch

- The command changes to exactly zero through the existing command setter.
- Environment info, actor observation, routing input, and teacher input must
  observe the same zero command before the next action.
- The transition row at the switch is retained and has transition_age=0.

### 3. Post-switch recovery phase

- The student continues generating states after the switch.
- The standing teacher is queried for action targets.
- The standing expert is the selected behavior-loss expert.
- Rows retain transition_age=0, 1, ..., recovery_horizon-1.
- Static standing rows and recovery standing rows share role=stand but retain
  different scenario labels.

### 4. DAgger outer-loop meaning

For student_k:

student_k
-> static stand and active walk rollout
-> walk_to_stop transition rollout
-> role-matched teacher relabel
-> cumulative aggregate of bootstrap and prior rounds
-> balanced update
-> student_(k+1)

The transition rollout must use the checkpoint produced by the previous outer
round. Increasing inner optimizer updates without a new student rollout is not
a new DAgger iteration.

## Proposed Row Contract

Every transition row must preserve:

| field | meaning |
| --- | --- |
| role | walk_flat before switch, stand after switch |
| scenario_label | static_stand, walk_flat, or walk_to_stop |
| transition_age | integer age after the zero-command switch; absent or -1 before switch |
| command_before | command used immediately before the switch |
| command_after | command visible to the row; zero for recovery rows |
| command_intent | active before switch, inactive after switch |
| student_obs | 98-D actor observation consumed by the student |
| teacher_obs | 98-D observation consumed by the selected teacher |
| teacher_action | detached 29-D action target from the selected teacher |
| source_checkpoint | exact teacher checkpoint identity |
| dagger_round | outer DAgger round that produced the row |

The save/load roundtrip must preserve all semantic fields. A shape-valid tensor
without scenario and transition identity is not a valid transition dataset.

## Routing, Loss, And Forbidden Behavior

Required:

- active -> walk expert 0 and walking teacher;
- inactive -> stand expert 1 and standing teacher;
- selected-expert behavior loss only updates the selected expert;
- transition rows remain distinguishable from static standing rows;
- the existing role and command-intent labels agree at every row.

Forbidden:

- adding a third transition expert;
- blending walk and stand actions without a new approved method contract;
- using standing targets for active pre-switch rows;
- using walking targets after the zero-command switch;
- collecting transition rows with a soft mixture while deployment uses hard routing;
- relabelling all standing rows as recovery rows;
- treating E21 teacher authority as proof of student recovery;
- treating action finiteness, checkpoint size, or offline MSE as physical acceptance.

## Proposed Workflow Integration

The formal single-entry workflow should:

1. Reuse compatible static stand and walk datasets through role preflight.
2. Collect only the transition scenario when its manifest is missing or stale.
3. Aggregate bootstrap, static DAgger, and transition DAgger sources
   cumulatively.
4. Record scenario counts, transition-age counts, teacher checkpoint hashes,
   parent student hash, and command schedule in the run manifest.
5. Resume completed outer rounds without silently repeating them.

Proposed initial batch quota:

- 50% walk_flat;
- 25% static_stand;
- 25% walk_to_stop recovery.

This is a starting acceptance quota, not a final research claim. The quota
must remain configurable and visible in the manifest.

## Acceptance Boundaries

| gate | required proof |
| --- | --- |
| S0 contract | human approves this proposal and the v002 registry transition |
| S1 schema | tiny golden rows preserve role, scenario, transition_age, commands, and targets through save/load |
| S2 workflow | one CPU toy round proves student_k -> transition rollout -> cumulative aggregate -> student_(k+1) |
| S4 live | student checkpoint is tested on repeated reset, forward/lateral/yaw command transitions, and walk-to-stop recovery |

The S4 minimum is not inherited from E21 alone. Promotion requires repeated
starts, at least 20 transitions across forward, lateral, and yaw commands, no
termination, task height/tilt limits, and velocity decay after stop.

## Human Approval Gate

Please approve or change these semantic decisions before activation:

1. Keep walk_to_stop as a scenario under Student-State DAgger, with no third
   expert.
2. Keep zero command as the only switch condition and the existing command
   thresholds unchanged.
3. Use the initial quota 50% walk_flat, 25% static_stand, 25% walk_to_stop.
4. Reuse existing static stand/walk datasets and collect only missing
   transition data.
5. Keep height control outside v002 until a qualified height teacher exists.

Until this gate is approved, this file remains a proposal and no transition
collector or training implementation should be activated.
