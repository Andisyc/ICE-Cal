# AMP-Only Async Walking Contract And Migration Proposal

Status: `proposal`

Date: 2026-07-21

This document is a replaceable Stage 1 proposal. It is not an active contract,
engineering authorization, or speedup claim.

## Figure/Contract Gate

Current status: `figure-mismatch`.

The active UniLab Concept Figure contains Teacher Policies, Command Intent,
Role Data, MoE Student, Robot Execution, and Student-State DAgger. AMP walking
is none of those objects. The AMP method therefore needs its own contract IDs
and Concept Figure; it must not reuse distillation IDs or be hidden below the
DAgger persistent runtime.

## Design Delta

Old design:

- UniLab G1 APPO trains command-following locomotion with the current task
  reward, including gait-phase terms.
- UniLab has no AMP discriminator, AMP expert sampler, AMP policy replay, or AMP
  transition payload in the async route.

New design:

- a walk-only expert transition distribution defines human walking style;
- the async collector produces policy transitions and exact AMP body-state
  transitions;
- the learner computes an AMP style reward, combines it with the minimal task
  reward, then performs V-trace/APPO and discriminator updates;
- the deployed actor has no discriminator or AMP expert-data dependency.

Changed semantic objects:

- `walk_expert_transition`;
- `policy_amp_transition`;
- `amp_style_reward`;
- `combined_training_reward`;
- `amp_discriminator_version`;
- `walk_only_dataset_manifest`.

Forbidden old assumptions:

- all files under `WalkandRun` are valid walking evidence;
- recovery or jogging data may enter through directory recursion;
- the collector should own discriminator training or synchronize discriminator
  weights;
- the existing fixed rollout schema can carry AMP transitions unchanged;
- an async/persistent process automatically proves end-to-end speedup;
- gait phase/contact rewards remain active because the base G1 task has them.

Expected runtime evidence:

- exact walk-only manifest and feature identity;
- AMP current/next transition lifecycle including termination and partial reset;
- discriminator version used to score every staged rollout batch;
- task/AMP/combined reward statistics and expert/policy logits;
- collector, IPC, learner, and discriminator timing split;
- stable checkpoint/resume identity;
- bounded 10-20 minute live outcome, reported as observed rather than inferred.

## Proposed Human Method Map

These are proposed Concept Figure blocks, not yet active IDs.

| Design ID | Canonical human name | Proposed contract section | Figure block | Current gap |
| --- | --- | --- | --- | --- |
| `AMP-WALK-DP-01` | Walk Expert Transitions | `AMP-WALK-METHOD-v001#walk-expert-transitions` | proposed `AW-M-01` | dataset filter/feature contract absent |
| `AMP-WALK-DP-02` | Policy Walk Transitions | `AMP-WALK-METHOD-v001#policy-walk-transitions` | proposed `AW-M-02` | async AMP payload absent |
| `AMP-WALK-DP-03` | AMP Style Discriminator | `AMP-WALK-METHOD-v001#amp-style-discriminator` | proposed `AW-M-03` | discriminator/normalizer/replay absent |
| `AMP-WALK-DP-04` | AMP-Regularized Walking Policy | `AMP-WALK-METHOD-v001#amp-regularized-policy` | proposed `AW-M-04` | reward-to-V-trace/update route absent |

The async collector, IPC schema, checkpoint, metrics, and tests are engineering
owners under these design points. They are not top-level method contributions.

## Proposed Method Closure

```text
walk-only expert transitions -------------------------+
                                                      |
policy body transitions -> AMP Style Discriminator -> amp_style_reward
policy actor/critic transitions -> minimal task reward -> combine
                                                      |
                                                      v
                                              V-trace + APPO
                                                      |
                                                      v
                                        human-like walking policy
```

## Recommended Async Semantic Order

Recommended decision:

```text
collector under actor/critic version k
 -> write actor, critic, action, behavior logp, task reward, AMP transition
 -> learner freezes discriminator D_k for the staged batch
 -> D_k scores every policy AMP transition exactly once
 -> combined reward enters V-trace
 -> APPO updates actor/critic
 -> discriminator update produces D_(k+1)
 -> publish new actor/critic weights
```

Rationale:

- the discriminator remains learner-only, so no discriminator weight sync or
  collector-side inference is added;
- all staged rollouts in one learner update have one explicit reward identity;
- V-trace sees the final combined reward rather than a post-update diagnostic;
- AMP non-stationarity is visible through discriminator-version metrics.

Rejected first-version alternatives:

- collector-side AMP inference: adds discriminator synchronization and GPU/CPU
  contention to the hot collector;
- synchronous PPO first: violates the requested async training target;
- scoring with `D_(k+1)` after discriminator update: changes the reward owner
  before the policy update and obscures batch identity;
- float16 AMP IPC in the correctness step: optimization before semantic parity.

## Semantic Source Of Truth

| Semantic object | Proposed active owner | Active consumers | Legacy path | Isolation rule | Implementation test | Integration test | Live gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| walk-only manifest | `unilab.algos.torch.amp.motion_dataset` | expert sampler | recursive AMP_mjlab directory scan | jog/idle/recovery patterns fail closed | manifest classification | formal compose/load | dataset quality |
| AMP state | G1 AMP env adapter | async payload, discriminator | AMP_mjlab mjlab observation manager | one 13-body/195D definition | coordinate/shape parity | reset/step route | sensor cost |
| policy AMP transition | AMP APPO collector payload | learner replay/scoring | AMP_mjlab runner-local transition | exact final observation on done | transition lifecycle | spawned collector IPC | real termination mix |
| AMP style reward | `AMPAPPOLearner.score_amp_reward` | V-trace reward owner, diagnostics | AMP_mjlab `predict_amp_reward` | one formula/version per batch | deterministic formula | reward-to-V-trace connectivity | learning scale |
| discriminator update | `AMPAPPOLearner.update_discriminator` | next batch reward owner | AMP_mjlab joint optimizer | separate learner optimizer | expert/policy loss | checkpoint/resume | stability |
| combined reward | AMP learner batch preparation | V-trace/APPO | current raw APPO task reward | task/AMP components logged separately | numeric composition | learner process order | policy quality |

## Scope

- MuJoCo G1 flat-ground walking first;
- one collector process and one learner process;
- 2048 env baseline before considering 4096;
- walk-forward/backward/sideways/arc expert clips only;
- full 13-body AMP state matching AMP_mjlab semantics;
- learner-side AMP scoring and discriminator update;
- APPO V-trace for stale behavior-policy correction;
- checkpoint/resume, metrics, normal play/export of the actor;
- no explicit gait observation or gait reward in the AMP task.

## Non-Scope

- jogging/running;
- fall recovery, delayed termination, or recovery reset;
- motion imitation/reset curriculum;
- Gait Phase, contact schedule, gait parameter fitting, or gait controller;
- Motrix in the first acceptance route;
- multi-GPU/multiple collectors;
- AMP at deployment;
- a promise that async execution alone achieves policy quality in 10-20 minutes.

## Provisional Step Map

This map becomes the engineering plan only after contract/Concept Figure
confirmation.

### Step 1 / 8: Baseline The Existing Async Route

Objective: establish whether current G1 APPO is collector-bound, learner-bound,
or IPC-bound on the target machine.

Scope: unchanged `g1_walk_flat/mujoco`, 2048 envs, bounded run, persisted timing.

Non-scope: AMP code or reward changes.

Owner files/modules: `scripts/train_appo.py`, `APPORunner`, `OffPolicyLogger`.

Expected evidence: S4/T-performance artifact with env steps/s, collector
inference/env-step time, learner time, wait fraction, H2D time, staging occupancy,
RSS/VRAM, and wall time.

Stop condition: one reproducible baseline and a named bottleneck. Stop/return to
design if the base async route cannot produce stable throughput evidence.

### Step 2 / 8: Add Reusable APPO Payload Extension Hooks

Objective: let algorithm-owned payloads extend APPO without copying the runner.

Scope: typed extra ring-buffer fields/dtypes, runner payload hooks, staging
support, memory budget, unchanged default/HORA behavior.

Non-scope: AMP semantics.

Owner files/modules: `ipc/rollout_ring_buffer.py`, `appo/{runner,worker,staging}.py`,
`ipc/memory_budget.py`.

Expected evidence: S1-S2/T-shape,T-connect,T-diff tests proving old APPO byte/
field behavior remains unchanged and optional fields survive spawned IPC.

Stop condition: a fake 195D payload crosses collector -> ring -> staging ->
learner with no runner fork and no regression.

### Step 3 / 8: Establish Walk Expert And AMP-State Owners

Objective: create one walk-only expert distribution and one AMP feature spec.

Scope: fail-closed manifest, 13-body ordering, vectorized cold-path conversion,
normalizer, feature parity with AMP_mjlab.

Non-scope: policy rollout or learning.

Owner files/modules: new `src/unilab/algos/torch/amp/{spec,motion_dataset}.py`,
motion assets/manifest, rotation helpers.

Expected evidence: S0-S1/T-shape,T-value,T-oracle tests; excluded clips cannot
enter; sampled transitions match the source transform within tolerance.

Stop condition: deterministic expert `(s_t,s_t+1)` sampling at 195D/195D.

### Step 4 / 8: Add AMP-Only G1 Environment Contract

Objective: emit exact policy AMP transitions without gait control.

Scope: opt-in tracked-body sensors, cached body IDs, `amp` observation, exact
terminal/final observation, partial reset, removal of gait-phase input/reward in
the new task only.

Non-scope: changes to existing `G1WalkFlat` semantics.

Owner files/modules: `envs/locomotion/g1/joystick.py`, backend contract calls,
new owner YAML.

Expected evidence: S1-S2/T-shape,T-order,T-diff reset/step/partial-reset tests;
existing task remains unchanged.

Stop condition: actor/critic/AMP groups and terminal transitions are correct for
full rollout and subset reset.

### Step 5 / 8: Implement AMP APPO Learner

Objective: make AMP reward and discriminator learning first-class learner owners.

Scope: discriminator, normalizer, policy replay, expert sampling, frozen-`D_k`
reward scoring, reward composition, discriminator optimizer, state dict.

Non-scope: collector-side discriminator or deployment inference.

Owner files/modules: new `amp/{discriminator,replay,appo_learner}.py`.

Expected evidence: S1-S2/T-value,T-order,T-persist tests for formula, one-score
per batch, V-trace connectivity, update order, and checkpoint round trip.

Stop condition: fake async batch changes V-trace returns through AMP reward and
advances exactly one discriminator version.

### Step 6 / 8: Connect The Formal Async Runtime

Objective: compose the AMP collector/learner through the existing APPO entrypoint.

Scope: AMP runtime resolver, runner hooks, AMP payload, checkpoint/logging,
Hydra task, one standard `uv run train --algo appo ...` route.

Non-scope: a second training script or copied lifecycle protocol.

Owner files/modules: new `amp/{appo_runner,appo_worker,runtime}.py`,
`scripts/train_appo.py` only through the existing resolver contract,
`conf/appo/task/g1_amp_walk/mujoco.yaml`.

Expected evidence: S2-S3/T-connect,T-order,T-persist spawned two-iteration run,
collector death propagation, resume, and actor-only playback.

Stop condition: formal command produces finite AMP/APPO metrics and a resumable
checkpoint without touching the legacy runner path.

### Step 7 / 8: Measure And Optimize The AMP Async Path

Objective: keep AMP overhead bounded relative to Step 1.

Scope: timing split, queue/staging behavior, memory budget; only after float32
correctness, evaluate float16 AMP transport or payload compaction.

Non-scope: reward/architecture tuning to hide throughput defects.

Owner files/modules: AMP runtime metrics, IPC dtype extension, benchmark artifact.

Expected evidence: S3-S4/T-performance,T-diff A/B artifact.

Stop condition: either AMP overhead is <=30% at matched env steps, or the exact
bottleneck and rejected 10-20 minute projection are reported.

### Step 8 / 8: Bounded 10-20 Minute Live Acceptance

Objective: test the user-facing wall-clock hypothesis and walking quality.

Scope: one frozen command/config/data/checkpoint identity, 10-20 minute budget,
postflight playback and AMP diagnostics.

Non-scope: running/recovery/gait control or indefinite tuning.

Owner files/modules: formal launch/postflight scripts, experiment tracker,
evidence ledger.

Expected evidence: S4/T-live,T-performance artifact containing wall time,
env steps, checkpoint hash, reward/logit curves, termination rate, and playback.

Stop condition: classify the result as runtime-pass/quality-pass, runtime-pass/
quality-fail, or performance-fail. A clean run is not automatically a useful
policy.

## Provisional S/T Acceptance Matrix

| Gate | Owner | S tier | T kinds | Required proof |
| --- | --- | --- | --- | --- |
| walk manifest | AMP dataset owner | S1 | T-oracle,T-value | no jog/idle/recovery paths |
| feature parity | AMP spec/env | S1-S2 | T-shape,T-value,T-diff | source and UniLab state agree |
| AMP IPC | ring/staging/collector | S1-S3 | T-shape,T-connect,T-order | exact current/next transition |
| reward route | AMP learner | S1-S3 | T-value,T-connect,T-order | AMP reward enters V-trace once |
| discriminator | AMP learner | S1-S2 | T-value,T-persist | finite loss/version/resume |
| legacy isolation | existing APPO/G1 | S2-S3 | T-diff | unchanged default task/runtime |
| async lifecycle | AsyncRunner | S2-S3 | T-order,T-persist | crash/close/cleanup/resume |
| performance | full route | S4 | T-performance | bounded overhead and timing owner |
| policy quality | full route | S4 | T-live | walking playback plus diagnostics |

## Work Estimate

- contract/Architecture activation: 1-2 days;
- baseline and reusable APPO extension seam: 2-4 days;
- AMP data/env semantics: 2-3 days;
- AMP async learner/runtime/checkpoint: 4-6 days;
- integration/regression tests: 2-4 days;
- performance/live gates: 2-4 days plus GPU runs.

Expected engineering total: roughly 13-21 working days for one engineer familiar
with UniLab. The resulting training run may be measured in minutes; the
migration and evidence work is not.

## Human Decision Required

Confirm or reject this one semantic decision:

> Keep the AMP discriminator learner-only. For each staged async batch, freeze
> `D_k`, compute the AMP reward before V-trace, update actor/critic with that
> reward, then train the discriminator to `D_(k+1)`.

Confirmation authorizes contract/Concept Figure creation and promotion of the
provisional step map. It does not authorize implementation or a live training
run.

