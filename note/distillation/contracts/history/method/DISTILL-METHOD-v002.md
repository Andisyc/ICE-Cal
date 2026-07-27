---
contract_id: DISTILL-METHOD-v002
status: superseded
effective_date: 2026-07-23
updated_date: 2026-07-27
supersedes: DISTILL-METHOD-v001
superseded_by: DISTILL-METHOD-v003
scope: G1 StandHeight and Walk two-teacher command-intent MoE distillation
concept_figure: note/architecture/concept/03_g1_multiteacher_distillation_method.data.json
---

# G1 StandHeight And Walk Two-Teacher Distillation Method Contract

## Problem

Stationary standing and height tracking are one supported behavior family: the
robot must track a requested base height while velocity intent remains zero.
Walking is a second behavior family with non-zero velocity intent and a fixed
nominal base height. The method must preserve both roles without averaging
their action targets into one undifferentiated policy.

## Design Point Register

| design_id | Canonical name | block_id | Contract section | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| DISTILL-DP-01 | Teacher Policies | DT-M-01 | [Teacher Policies](#teacher-policies) | Remote teacher identities and role artifacts are recorded; bytes were not reread in the local repair task |
| DISTILL-DP-02 | Command Intent | DT-M-02 | [Command Intent](#command-intent) | Deterministic 3x3 transition routing passed; retrained live transitions are unproven |
| DISTILL-DP-03 | Role Data | DT-M-03 | [Role Data](#role-data) | 99-D schema, roundtrip, legacy isolation, and per-case transition metadata passed locally |
| DISTILL-DP-04 | MoE Student | DT-M-04 | [MoE Student](#moe-student) | Round-2 student exists but failed the governed non-nominal transition gate |
| DISTILL-DP-05 | Student-State DAgger | DT-M-05 | [Student-State DAgger](#student-state-dagger) | Repaired collection distribution has not run on SSH |

## Teacher Policies

- `design_id`: `DISTILL-DP-01`
- `block_id`: `DT-M-01`
- Meaning: exactly two qualified teacher actors provide detached action targets.
- `StandHeight`: velocity command is zero; target base height varies; the actor
  must preserve upright posture, double-foot support, and low foot slip while
  tracking height.
- `Walk`: any active planar or yaw velocity command selects walking; target
  height is forced to the nominal `0.754 m` and does not define a third role.
- Input: a common 99-D actor observation, equal to the legacy 98-D G1 actor
  observation with `target_height` inserted at command index 96.
- Output: detached 29-D teacher actions plus exact task, config, checkpoint,
  adapter, and checkpoint-hash identity.
- Warm-start rule: a legacy 98-D actor may initialize a 99-D actor only through
  an explicit actor-only adapter that inserts a zero first-layer input column at
  index 96 and migrates normalizer statistics. Critics, target critics, replay,
  optimizer state, and old artifacts are not migrated.
- Curriculum: first prove equivalence at fixed `0.754 m`; then train
  `[0.65, 0.754] m`; expand toward `0.50 m` only after the preceding quality
  gate passes.
- Forbidden: treating a `G1WalkFlat` checkpoint or the existing student
  checkpoint as a height teacher; starting the first curriculum at `0.30 m`;
  using a teacher before checkpoint identity and physical quality are recorded.

## Command Intent

- `design_id`: `DISTILL-DP-02`
- `block_id`: `DT-M-02`
- Meaning: the existing explicit command thresholds classify velocity intent.
- `inactive`: route to `StandHeight` / expert 1, force velocity command to zero,
  and preserve the requested target height.
- `active`: route to `Walk` / expert 0 and force target height to `0.754 m`.
- Ownership: task and collection config own thresholds and height semantics;
  checkpoint filenames and a free learned router do not.
- Interaction: the same intent mapping labels Role Data and selects collection,
  behavior-loss, DAgger rollout, and deployment experts.
- Forbidden: creating a third height intent, changing thresholds between
  collection and deployment, or allowing active walking rows to retain an
  arbitrary target height.

## Role Data

- `design_id`: `DISTILL-DP-03`
- `block_id`: `DT-M-03`
- Meaning: every training row preserves the behavior and command semantics that
  produced its teacher target.
- Required row contract: 99-D `student_obs`, 99-D `teacher_obs`, detached 29-D
  `teacher_action`, `[vx, vy, vyaw]` command, scalar `target_height`,
  `active|inactive` intent, `walk|stand_height` role, and source identity.
- Output: a `DistillationTensorDataset` whose role, intent, scenario, and source
  metadata survive save, load, aggregation, resume, and fork.
- Legacy isolation: existing 98-D stand/walk datasets and checkpoints remain
  immutable legacy artifacts. They must be explicitly adapted and recollected;
  mixed 98-D and 99-D source rows fail closed.
- Forbidden: inferring height from base state instead of retaining the command,
  padding arbitrary source observations, or claiming quality from row count.

## MoE Student

- `design_id`: `DISTILL-DP-04`
- `block_id`: `DT-M-04`
- External execution block: `DT-X-01`.
- Meaning: one 99-D checkpoint contains exactly two behavior-specialized
  experts and a routing mechanism.
- Input: 99-D actor observation including target height at index 96.
- Output: one 29-D action.
- Mapping: `walk/active -> expert 0`; `stand_height/inactive -> expert 1`.
- Training rule: behavior cloning updates only the selected expert; inactive
  expert parameters and optimizer state must not move.
- Required behavior: zero velocity tracks the requested height; any active
  velocity command walks at nominal height; returning to zero velocity recovers
  `StandHeight` and preserves the requested post-transition height.
- Forbidden: a third expert without a new contract version, soft-mixture
  collection presented as pure-expert evidence, dimension-tolerant checkpoint
  loading, or finite action used as physical acceptance.

## Student-State DAgger

- `design_id`: `DISTILL-DP-05`
- `block_id`: `DT-M-05`
- Meaning: close the behavior-cloning distribution gap by querying the matching
  teacher on states generated by the selected student expert.
- Outer loop: `student_k rollout -> matching teacher relabel -> aggregate 1..k
  -> selected expert/router update -> student_(k+1)`.
- Formal workflow, artifact reuse, resume, fork, and persistent-runtime behavior
  remain owned by `DISTILL-TRAIN-v003`; its default remains `legacy`.
- Required scenarios: low/mid/high static heights, StandHeight-to-Walk,
  Walk-to-StandHeight at nominal height, Walk-to-StandHeight at a non-nominal
  height, repeated reset, and bounded long-horizon support/tilt/termination.
- Forbidden: using one fixed dataset as multiple DAgger iterations, relabeling
  with a non-matching role teacher, or promoting a connector-valid checkpoint
  without physical acceptance.

## Semantic Ownership And Migration

| Semantic object | Active owner | Active consumers | Legacy isolation |
| --- | --- | --- | --- |
| `target_height` | G1 command config and reset/resample owner | observation, reward, StandHeight teacher/student | `G1StandStill` remains 98-D |
| Dynamic standing target | G1 reward context and standing reward helpers | StandHeight reward stack | fixed scalar fallback preserves legacy |
| 98-D to 99-D actor migration | off-policy checkpoint adapter | StandHeight warm start and converted Walk teacher | source checkpoints are immutable |
| Two teacher roles | new distill workflow owner YAML | collector, manifest, DAgger, trainer | old `stand`/`walk_flat` profile is unchanged |
| Two expert mapping | new distill workflow owner YAML | trainer and playback | old three-expert checkpoint/config is unchanged |

## Required Evidence

1. S0/S1: Hydra isolation, observation shape, dynamic target reward ordering,
   and unchanged legacy task contracts.
2. S1: deterministic actor-output equivalence for the 98-D to 99-D adapter,
   fail-closed incompatible checkpoint handling, and persisted migration hashes.
3. S2/S3: common 99-D role dataset roundtrip, selected-expert update isolation,
   strict two-expert checkpoint reload, and formal route connectivity.
4. S4 teacher gate: low/mid/high target ordering, per-bin mean absolute height
   error at most `0.05 m`, no bounded-window termination, double-foot support
   fraction at least `0.90`, and tilt below the task limit.
5. S4 student gate: repeated reset and both transition directions, including a
   non-nominal post-walk height target, with exact checkpoint and config identity.

## Current Acceptance Status

The method semantics are active, and Steps 2-4 remain complete at their
deterministic boundaries. E114 preserves the retained Step 2 result (`108
passed, 24 warnings in 19.46s`) and Step 3 result (`8 passed in 6.77s`); E113
records Step 4 Ruff PASS and `27 passed in 20.56s`. E115 confirms the bounded
one-environment `G1StandHeight` route, and E116 confirms async/double-buffer and
explicit `persistent_async` connectors while retaining the governed `legacy`
default and E67 `NO_STABLE_SPEEDUP` limitation. E117 records the current
round-2 99-D student failure and a deterministic command x recovery-height
DAgger distribution repair (`10 + 7` focused tests and Ruff PASS). Remote
teacher, dataset, and student identities are conversation-backed; their bytes
were not read locally. The repaired fork, repeated-reset transition behavior,
and final policy-quality acceptance remain unexecuted.
