# G1 Stand-Still Expert Plan

## Problem

Pure upstream-style `G1WalkFlat` can train a normal walking policy after height tracking is disabled. This is new runtime evidence that the height-conditioned route can conflict with the original walking reward/distribution.

The next goal is not to repair Walking by adding another mixed reward. The next goal is to train a separate static standing expert, then later distill multiple experts.

## Concept Boundary

Current concept variable: static balance under zero external velocity command.

Inside scope:
- zero-command standing;
- stable upright posture;
- centered base over feet;
- both feet in contact with balanced support footprint;
- no sliding of feet while they are in contact;
- fixed base height target as a posture penalty;
- small corrective actions for balance;
- recovery from small sampled initial velocity/tilt perturbations after the static policy is stable.

Outside scope:
- height-command tracking;
- walking gait phase tracking;
- Standing -> Walking transition;
- Walking -> Standing transition;
- mode-conditioned routing inside one actor;
- MoE/student distillation implementation.

## Current Evidence

Hydra probe, 2026-07-09:

| Case | Observed contract |
| --- | --- |
| `task=sac/g1_walk_flat/mujoco` | `task_name=G1WalkFlat`, no `mode_observation`, no `reward.mode`, no `gait_constraint`, no height command, reward keys are upstream walking keys. |
| `task=sac/g1_walk_flat/mujoco +g1_walk_stage=standing_sanity` | Still `task_name=G1WalkFlat`, enables `mode_observation=true`, `reward.mode`, `gait_constraint`, zero commands, and both stand and walk term lists. |
| `task=sac/g1_walk_height/mujoco` | `task_name=G1WalkHeight`, no mode route, adds `observe_height_command=true` and `track_base_height_exp_smooth`. |

Implication: `standing_sanity` is useful as a previous experiment, but it is not clean enough as the owner route for a standalone standing expert.

## Feature-Flag / Owner-Task Contract

Owner route:
- Add `task=sac/g1_stand_still/mujoco`.
- Add `G1StandStillCfg`, registered as `G1StandStill`, reusing `G1WalkEnv`.

OFF behavior:
- `task=sac/g1_walk_flat/mujoco` remains 98-dim upstream-style Walking.
- No standing reward keys in default Walking.
- No `mode_observation`.
- No `reward.mode`.
- No `reward.gait_constraint`.
- No height command observation.

ON behavior:
- `training.task_name=G1StandStill`.
- Commands are always zero.
- Actor obs remains 98-dim: no mode obs and no height command.
- Reward is direct dispatch of standing/static balance terms, not `reward.mode`.
- No walking gait rewards: no `tracking_lin_vel`, `tracking_ang_vel`, `feet_phase`, `feet_phase_contrast`, `feet_phase_contact`.
- No height-command tracking: no `observe_height_command`, no `track_base_height_exp_smooth`.
- `stand_action_authority=false`, so the actor can use small corrective actions.
- The reward must not penalize absolute nonzero action; static standing on this G1 model requires a nonzero support action.

Forbidden mixed states:
- `G1StandStill` + `observe_height_command=true`.
- `G1StandStill` + `reward.mode.enabled=true`.
- `G1StandStill` + `reward.gait_constraint.enabled=true`.
- `G1StandStill` + any walking velocity tracking reward.
- `G1WalkFlat` default gaining standing reward keys.

## Parameter Inventory

| Param | Owner | OFF value | ON value | Consumers | Risk |
| --- | --- | --- | --- | --- | --- |
| `training.task_name` | task yaml | `G1WalkFlat` | `G1StandStill` | registry/env creation, run_config, playback | checkpoint route must match task. |
| `env.commands.vel_limit` | task yaml | walking range | `[[0,0,0],[0,0,0]]` | command sampler, obs, gait mask | command must force static role. |
| `env.commands.rel_standing_envs` | task yaml | absent/default | `1.0` | command sampler/log role | redundant but documents role. |
| `env.commands.rel_transition_envs` | task yaml | absent/default | `0.0` | command sampler | no transition distribution. |
| `env.mode_observation` | env cfg | absent/false | absent/false | obs dim, symmetry, checkpoint | keep 98-dim for later distillation with Walking. |
| `env.commands.observe_height_command` | command cfg | absent/false | absent/false | obs dim, symmetry, checkpoint | avoid height conflict. |
| `reward.scales` | task yaml | walking terms | standing terms only | reward dispatch | main objective definition. |
| `reward.mode` | reward cfg | absent/default false | absent/default false | reward dispatch | no mode routing. |
| `reward.gait_constraint` | reward cfg | absent/default false | absent/default false | gait phase/action/reward bridge | no gait contamination. |
| `reward.base_height_target` | reward cfg | `0.754` | `0.754` initially | fixed posture penalty | fixed target, not command input. |
| `env.stand_action_authority` | env cfg | false | false | `apply_action` | actor must be able to balance. |
| `env.reset_base_qvel_limit` | env cfg | `0.5` | stage schedule: `0.0 -> 0.2/0.5` | reset DR | split static stand and recovery stand stages. |

## Reward Shape

Initial static expert reward should contain posture/stability terms only:

```text
 upright
+ alive
- penalty_orientation
- penalty_ang_vel_xy
- penalty_action_rate
- base_height
- pose
- penalty_feet_ori
- stand_still
- stand_dof_vel_l2
- stand_lin_vel_xy_l2
- stand_yaw_vel_l2
- stand_tilt_l2
- stand_tilt_margin_l2
- stand_fall_l2
- stand_both_feet_contact
- stand_foot_contact_balance
- stand_feet_slide_l2
- stand_feet_x_l2
- stand_feet_y_width_l2
- stand_feet_yaw_l2
- stand_base_feet_center_x_l2
- stand_base_feet_center_y_l2
```

Do not include:

```text
tracking_lin_vel
tracking_ang_vel
feet_phase
feet_phase_contrast
feet_phase_contact
track_base_height_exp_smooth
reward.mode
reward.gait_constraint
```

Rationale: static standing and walking are different policy manifolds. Shared terms can exist only when they describe posture or safety, not when they imply gait/action timing.

Update, 2026-07-09: agile-demo standing behavior confirms that pose shaping alone is not the full static standing objective. The migrated standalone expert now adds agile-style support terms: both-feet contact, contact-count balance, and contacted-foot slide penalty. `stand_action_l2` is intentionally absent from the standalone `G1StandStill` owner YAML because live anchor search showed the stable static support action is nonzero.

Live sentinel after migration, 2026-07-09:
- `G1StandStillCfg` uses `scene_flat.xml` plus `stand_support_task.xml` so the owner scene exposes `left_foot_linvel/right_foot_linvel` without changing default `G1WalkFlat`.
- Command/max abs is `0.0`, `gait_enabled_mean=0.0`, obs spec remains `obs=98`, and forbidden walking/height reward keys are absent.
- 16-step MuJoCo rollout with zero action had `terminated_total=0`, final tilt max `4.75 deg`, base height min `0.7358`, both-feet contact mean `1.0`, contact balance mean `0.0`, and contacted-foot slide mean `8.2e-7`.
- 64-step MuJoCo zero-action rollout still fails stability gates: tilt max `43.09 deg`, base-over-feet x range `[0.4501, 0.4961]`, base height min `0.5669`, both-feet contact mean `0.0`, and contact balance mean `1.0`.
- 64-step static-anchor search finds a nonzero lower-body constant action with tilt `3.43 deg`, base height `0.6598`, base-over-feet x `-0.0254`, and no termination; zero action in the same probe has tilt `29.39 deg`, base height `0.5680`, base-over-feet x `0.4948`.
- This validates the support-sensor/reward plumbing and the existence of a learnable nonzero static support action. It does not prove a newly trained stand-still policy quality yet.

## Execution Steps

### Step 1: Branch Isolation And Contract Note

Scope: create an isolated branch and keep this plan as the source-of-truth.

Command:

```bash
git switch -c codex/g1-stand-still-expert
git status --short --branch
```

Stop condition: branch name is visible and worktree only contains intended files.

### Step 2: Config-Only Owner Route

Scope: add `G1StandStillCfg` registration and `conf/offpolicy/task/sac/g1_stand_still/mujoco.yaml`.

Non-scope: do not tune weights beyond carrying current standing terms into a clean direct-dispatch task.

Tests:

```bash
uv run pytest tests/config/test_reward_injection.py -q
uv run pytest tests/envs/locomotion/g1/test_symmetry_contract.py -q
```

Expected facts:
- flat OFF path unchanged;
- stand-still ON path has 98 actor obs dim;
- no mode, no height, no gait constraint;
- command distribution is zero-only.

Status, 2026-07-09:
- Implemented `G1StandStillCfg` and `task=sac/g1_stand_still/mujoco`.
- Added config contract test and symmetry/obs-dim contract test.
- Evidence is offline contract only; live standing stability remains Step 4.

### Step 3: Standing Reward Lab Contract Tests

Scope: add S1/S2 tests that compare clean stand, low crouch, rear-lean, wide feet, x-offset feet, yawed feet, and moving base samples.

Test class: module semantic + offline connectivity.

Expected facts:
- clean stand scores highest;
- base behind feet is penalized;
- foot x mismatch is penalized;
- inward/outward foot yaw is penalized;
- moving base is penalized even with zero command;
- no walking/gait term can contribute.

Status, 2026-07-09:
- Added `test_g1_stand_still_reward_lab_prefers_clean_stand_over_pose_failures`.
- The lab composes `task=sac/g1_stand_still/mujoco`, builds labeled fake-backend rows for clean stand, rear lean, wide feet, staggered feet, toe-in, low crouch, and moving base, then asserts clean stand has the highest weighted reward.
- The lab also asserts walking, gait, and height-command reward terms are absent from the stand-still reward scale set.
- Evidence is S1/S2 offline reward connectivity only; Step 4 live sentinel is still required before claiming simulator stability.

### Step 4: Live Sentinel Before Long Training

Scope: add or reuse a no-viewer live probe for `G1StandStill`.

Command shape:

```bash
uv run scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py --num-envs 16 --steps 64
```

Required printed facts:
- resolved task and reward keys;
- actor obs dim and command dim;
- command histogram, all zero;
- gait_enabled fraction, expected 0.0;
- base height, tilt, foot width, foot x offset, base-over-feet offset;
- first-step raw action and executed action stats;
- per-term reward means;
- termination count and first termination reason.

Stop condition: real env reaches 64 steps without immediate collapse under zero/random-small policy probe, or the first failing boundary is named.

Status, 2026-07-09:
- Added `scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py`.
- The sentinel composes `task=sac/g1_stand_still/mujoco`, creates the real MuJoCo env, disables autoreset, runs zero-action rollout, and prints reset/rollout base height, tilt, foot width, base-over-feet offset, termination reason, and per-term reward contributions.
- `uv run scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py --num-envs 16 --steps 64 --seed 11` reached 64 steps with no termination, but failed stability gates: final tilt max `43.09 deg`, base-over-feet x range `[0.4501, 0.4961]`, and base height min `0.5669`.
- First named live boundary: zero-action stand-still reset drifts into rear-lean / crouch near-fall even though command routing and reward-key isolation are clean.
- Evidence is S4 live sentinel for zero-action dynamics, not trained-policy stability.

Posture-anchor update, 2026-07-09:
- Locked the standalone stand-still posture anchor to the upstream `G1WalkFlat` shaping contract: `penalty_orientation`, `penalty_ang_vel_xy`, `penalty_action_rate`, `pose`, `penalty_feet_ori`, `alive`, `base_height_target`, `min_base_height`, `max_tilt_deg`, and `pose_weights`.
- Added `test_offpolicy_g1_stand_still_pose_anchor_matches_upstream_walking_reward`.
- Re-ran the 64-step live sentinel after the anchor contract. The failure boundary did not change: final tilt max `43.09 deg`, base-over-feet x range `[0.4501, 0.4961]`, base height min `0.5669`.
- Implication: the default posture reward is now explicitly upstream-aligned; the remaining issue is a physical zero-action/static-action anchor problem, not a missing upstream posture-shaping term.

Static-action anchor update, 2026-07-09:
- Extended the live sentinel with module probes based on the atlas rows `G1LOC-C-004`, `G1LOC-ENV-001`, `G1LOC-ACT-001`, and `G1LOC-STAND-001`.
- `--probe-mode anchor-diff` showed `reset_dof_pos == default_angles`, base qvel is exactly zero, commands are exactly zero, gait is disabled, and `hold_reset_pose` is identical to `zero_action`. Both fail the 64-step stability gate.
- `--probe-mode anchor-search` searched a constant lower-body action over the first 12 joints and found a nonzero static anchor with action L2 `0.3100`, final tilt `3.43 deg`, base height `0.6598`, and base-over-feet x `-0.0254`, versus zero-action final tilt `29.39 deg`, base height `0.5680`, and base-over-feet x `0.4948`.
- Boundary decision: the reset/default posture is aligned but not passively stable under zero residual action. A physically useful nonzero static action anchor exists, so the Standing design needs an explicit static support-action anchor or reward formulation that does not force the actor back toward zero/default when balance requires residual leg action.

### Step 5: Short Training Probe

Scope: train a tiny stand-still run with live diagnostics enabled.

Command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> HYDRA_FULL_ERROR=1 PYTHONWARNINGS="ignore" uv run train \
  --algo sac \
  --task g1_stand_still \
  --sim mujoco \
  algo.max_iterations=200 \
  algo.save_interval=200
```

Expected facts:
- no obs dim mismatch;
- reward log shows standing terms only;
- episode length increases or at least does not immediately collapse;
- no walking/height reward appears in logs.

### Step 6: Full Standing Training

Scope: run the full standing expert after Step 5 passes.

Command shape:

```bash
CUDA_VISIBLE_DEVICES=<gpu> HYDRA_FULL_ERROR=1 PYTHONWARNINGS="ignore" nohup uv run train \
  --algo sac \
  --task g1_stand_still \
  --sim mujoco \
  > train_g1_stand_still_sac_mujoco.txt 2>&1 &
```

### Step 7: Distillation Later

Do not start distillation until standalone experts are usable:
- Walking expert: upstream-style `G1WalkFlat`.
- Standing expert: `G1StandStill`.
- Optional future expert: height-conditioned or recovery expert, only after conflict is understood.

Distillation should be a separate branch and contract:
- teacher checkpoint registry;
- teacher obs normalization contract;
- command/state-based router or MoE gate;
- student obs/action dim contract;
- teacher action target or Q-weighted imitation objective;
- playback route proving the student, not teachers, is deployed.

## Test Matrix

| Module | Owner | Required S/T | Evidence target |
| --- | --- | --- | --- |
| Stand-still task config | `conf/offpolicy/task/sac/g1_stand_still/mujoco.yaml` | S0/S2, T-connect, T-oracle | ON/OFF config purity. |
| Stand-still env registration | `src/unilab/envs/locomotion/g1/joystick.py` | S0/S2, T-connect | `G1StandStill` resolves to `G1WalkEnv`. |
| Observation layout | `G1WalkEnv.obs_groups_spec`, symmetry | S1/S2/S3, T-shape, T-order, T-persist | 98-dim actor obs, no mode/height. |
| Command distribution | `sample_g1_walk_commands` | S1/S2/S4, T-dist, T-role, T-live | all commands zero; gait mask false. |
| Standing reward | G1 stand reward functions | S1/S2/S4, T-value, T-role, T-oracle, T-live | clean stand outranks bad stand; live per-term snapshot. |
| Action execution | `G1WalkEnv.apply_action` | S1/S2/S4, T-transform, T-scale, T-live | actor action is not zeroed in static stand. |
| Checkpoint/playback | train run_config, play scripts | S3/S4, T-persist, T-diff, T-live | checkpoint obs dim matches playback env. |

## Next Safest Action

Next safest action: add a contract-tested static support-action anchor path before long training. The current live sentinel says the upstream-aligned default pose is not zero-action stable, while a nonzero lower-body constant action can stabilize the robot for the same 64-step probe.
