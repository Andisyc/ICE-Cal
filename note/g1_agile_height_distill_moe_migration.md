# G1 Agile Height, Distillation, and MoE Migration Plan

Date: 2026-07-08

Status: planning note. Evidence is code-confirmed from UniLab CodeGraph, Agile CodeGraph, local source search, and web repository search. No UniLab code has been changed by this note.

## Problem

UniLab G1 currently needs a cleaner path for separating standing, walking, and recovery behaviors without repeatedly reshaping rewards by visual feedback alone.

Agile-demo already contains two useful mechanisms:

1. Height-conditioned velocity tracking for G1 locomotion.
2. Teacher-student distillation for G1 velocity-height policies.

The intended migration order is:

1. Port height tracking first.
2. Port G1 distillation second.
3. Add MoE student distillation last, using external MoE code only as a reference pattern.

## Non-Scope

- Do not replace the existing UniLab G1 walking reward stack in the first step.
- Do not introduce MoE before height command, height observation, and height reward contracts are stable.
- Do not directly copy GPL code into UniLab.
- Do not claim policy quality from contract tests; simulator behavior still needs live sentinels.

## Source Evidence

### UniLab owner paths

- Command schema and shared command helpers: `src/unilab/envs/locomotion/common/commands.py`
- Height reward primitive: `src/unilab/envs/locomotion/common/rewards.py`
- Locomotion base env and action scaling: `src/unilab/envs/locomotion/common/base.py`
- G1 base config: `src/unilab/envs/locomotion/g1/base.py`
- G1 walk command sampling, obs, reward dispatch, and reset lifecycle owner: `src/unilab/envs/locomotion/g1/joystick.py`
- Existing G1 configs: `conf/*/task/g1_walk_flat/*`
- Existing distillation code is HORA/Sharpa-specific, not G1 locomotion-general: `scripts/train_hora_distill.py`, `src/unilab/algos/torch/hora/distill.py`

### Agile owner paths

- Height command and reward task config: `/Users/chengyuxuan/ArtiIntComVis/agile-demo/WBC-AGILE/agile/rl_env/tasks/locomotion_height/g1/velocity_height_env_cfg.py`
- Height reward function: `/Users/chengyuxuan/ArtiIntComVis/agile-demo/WBC-AGILE/agile/rl_env/mdp/rewards/task_rewards.py`
- Distillation runner config: `/Users/chengyuxuan/ArtiIntComVis/agile-demo/WBC-AGILE/agile/rl_env/tasks/locomotion_height/g1/agents/rsl_rl_ppo_cfg.py`
- Distillation algorithm: `/Users/chengyuxuan/ArtiIntComVis/agile-demo/WBC-AGILE/agile/algorithms/rsl_rl/rsl_rl/algorithms/distillation.py`
- Student models: `/Users/chengyuxuan/ArtiIntComVis/agile-demo/WBC-AGILE/agile/algorithms/rsl_rl/rsl_rl/modules/student_trained_teacher.py`, `/Users/chengyuxuan/ArtiIntComVis/agile-demo/WBC-AGILE/agile/algorithms/rsl_rl/rsl_rl/modules/student_trained_teacher_recurrent.py`

### External MoE references

- `lucidrains/mixture-of-experts`: MIT, PyTorch top-2 MoE with capacity factor and auxiliary balancing loss. Reference: https://github.com/lucidrains/mixture-of-experts
- `davidmrau/mixture-of-experts`: useful readable PyTorch sparse dispatcher pattern, but GPL-3. Do not copy code into UniLab. Reference: https://github.com/davidmrau/mixture-of-experts
- `n3il666/Meta-DMoE`: MIT, multi-expert distillation framing with separately trained experts. Reference: https://github.com/n3il666/Meta-DMoE
- `SimiaoZuo/MoEBERT`: MoE via distillation/adaptation, useful as conceptual reference for upcycling dense policies into experts. Reference: https://github.com/SimiaoZuo/MoEBERT
- `microsoft/Tutel`: optimized MoE library, useful for terminology and future scale, too heavy for first robotics prototype. Reference: https://github.com/microsoft/tutel

## Migration Principle

The core variable is not "more network capacity". The core variable is conditional behavior separation:

```text
commanded height / velocity / recovery state
  -> route to the right behavioral manifold
  -> imitate or optimize the action distribution for that manifold
  -> deploy with only policy-safe observations
```

This means MoE must be introduced after the signals that define the manifolds are stable. Otherwise, the router has no reliable evidence and will collapse or learn arbitrary shortcuts.

## Phase 1: Migrate Agile Height Tracking

### Phase 1 Boundary

Scope: make "target base height" an explicit, testable G1 command/reward variable in UniLab.

Non-scope:

- no MoE;
- no teacher-student distillation;
- no checkpoint format changes;
- no silent observation-dimension change to the existing `g1_walk_flat` task;
- no broad standing/walking reward retune.

Concept variable:

```text
target_base_height
  -> observable command or side-channel
  -> measured base/pelvis height
  -> reward ordering
  -> optional policy/critic observation contract
```

The first migration must preserve semantic strategy transfer over module copy-paste. Agile's useful idea is not its IsaacLab class hierarchy; it is the closed loop:

```text
sample height target
  -> measure current height
  -> reward exp(-height_error / std^2)
  -> expose height intent to the policy only through a deliberate obs contract
```

### Phase 1 Step Breakdown

Each step below should be executable by one focused coding pass. A step should not start until the previous stop condition is met.

Step sizing rationale:

| Step | Boundary type | Why this is the right size |
| --- | --- | --- |
| 1.1 | Pure command helper | Can be proven with deterministic arrays before env wiring. |
| 1.2 | Pure reward helper | Reward ordering can be proven without simulator startup. |
| 1.3 | Measurement accessor | Isolates backend z-height semantics from reward math. |
| 1.4 | RewardContext connector | Crosses one connector only: `info`/cfg target -> reward context. |
| 1.5 | Config owner | Protects old checkpoints before obs dimensions change. |
| 1.6 | Observation shape contract | Deliberately crosses the fragile policy I/O boundary. |
| 1.7 | Live sentinel | Only after local contracts pass, crosses reset/step/log lifecycle once. |

### Step 1.1: Height Command Data Contract

Scope: add pure command helpers and config fields for target height without touching G1 env runtime.

Non-scope: no reward, no obs, no reset lifecycle, no YAML task registration.

Files:

- Modify: `src/unilab/envs/locomotion/common/commands.py`
  - Add height command fields to `Commands` or a small derived config used by G1 height tasks.
  - Add a pure helper for sampling target heights with deterministic low/default/high cases.
- Create: `tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`
  - Add command-only tests first.

Owner module: `common/commands.py`.

Core parameter path:

```text
height_range/default_height/random_height_during_walking
  -> sample_height_commands(...)
  -> height_cmd array with shape (N, 1)
```

Test class: core param path.

Command:

```bash
uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k height_command
```

Expected result:

- PASS after implementation.
- Before implementation, expected failure is an import or missing-symbol failure for the height helper.

Probe/assert facts:

```text
height_cmd.shape == (num_samples, 1)
height_cmd.dtype is global dtype
height_cmd.min() >= height_range[0]
height_cmd.max() <= height_range[1]
default-only sample equals default_height
low/default/high fixture preserves ordering
```

Stop condition:

- The helper has a semantic toy fixture where low/default/high target heights are generated deterministically and checked by assertions.

Evidence (2026-07-08):

- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k height_command`: PASS (`3 passed`).
- `uv run python -m py_compile src/unilab/envs/locomotion/common/commands.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`: PASS.

### Step 1.2: Height Reward Formula Contract

Scope: add an Agile-equivalent smooth height tracking reward in the shared reward layer.

Non-scope: no G1 reward dispatch wiring, no config changes, no policy obs changes.

Files:

- Modify: `src/unilab/envs/locomotion/common/rewards.py`
  - Add a positive reward helper equivalent to Agile's `track_base_height_exp_smooth`.
  - Keep existing `base_height` penalty intact for backward compatibility.
- Modify: `tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`
  - Add reward-ordering tests.

Owner module: `common/rewards.py`.

Core parameter path:

```text
RewardContext.base_height
RewardContext.base_height_target or per-env target height
  -> height_error
  -> exp reward in [0, 1]
```

Test class: tiny golden fixture.

Command:

```bash
uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k height_reward
```

Expected result:

- PASS after implementation.
- Target-height sample reward is greater than low/high off-target rewards.
- Symmetric errors around target produce equal reward.

Probe/assert facts:

```text
reward_at_target == 1.0 within tolerance
reward_low < reward_at_target
reward_high < reward_at_target
reward_low == reward_high for equal absolute error
all rewards finite
```

Stop condition:

- The reward formula is locally proven without constructing a simulator env.

Evidence (2026-07-08):

- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k height_reward`: PASS (`2 passed, 3 deselected`).
- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q`: PASS (`5 passed`).
- `uv run python -m py_compile src/unilab/envs/locomotion/common/rewards.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`: PASS.
- `uv run ruff check src/unilab/envs/locomotion/common/rewards.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`: PASS.

### Step 1.3: G1 Runtime Measurement Contract

Scope: prove UniLab G1 already has a stable runtime measurement for current base height and define the owner accessor if needed.

Non-scope: no command concat, no obs dim change, no reward dispatch change.

Files:

- Modify only if needed: `src/unilab/envs/locomotion/g1/joystick.py`
  - Reuse or narrow `_terrain_relative_base_height()`.
- Test: `tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`
  - Use a fake backend or fake env object if a full MuJoCo env is not required.

Owner module: `g1/joystick.py`.

Core parameter path:

```text
backend.get_base_pos()[:, 2]
  -> _terrain_relative_base_height()
  -> measured_height shape (N,)
```

Test class: secondary contract path.

Command:

```bash
uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k measured_height
```

Expected result:

- PASS with a fake backend returning a known `(N, 3)` base position array.
- The measured height is exactly the z column.

Probe/assert facts:

```text
base_pos.shape == (N, 3)
measured_height.shape == (N,)
measured_height equals base_pos[:, 2]
measured_height dtype is global dtype
```

Stop condition:

- The measured-height boundary is proven independently of reward and observation wiring.

Evidence (2026-07-08):

- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k measured_height`: PASS (`1 passed, 5 deselected`).
- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q`: PASS (`6 passed`).
- `uv run python -m py_compile tests/envs/locomotion/g1/test_g1_height_tracking_contract.py src/unilab/envs/locomotion/g1/joystick.py`: PASS.
- `uv run ruff check tests/envs/locomotion/g1/test_g1_height_tracking_contract.py src/unilab/envs/locomotion/g1/joystick.py`: PASS.

### Step 1.4: Height Target Bridge Into RewardContext

Scope: connect sampled target height to `RewardContext` without changing actor observation dimensions.

Non-scope: no policy obs height command, no new task registration, no MoE/distill.

Files:

- Modify: `src/unilab/envs/locomotion/common/rewards.py`
  - If needed, extend `RewardContext` with a per-env height target field while preserving scalar `base_height_target` fallback.
- Modify: `src/unilab/envs/locomotion/g1/joystick.py`
  - In `_build_reward_context`, read height target from `info` when present.
- Modify: `tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`
  - Add fake-context tests for scalar fallback and per-env override.

Owner module: `g1/joystick.py` as connector, `common/rewards.py` as data contract.

Core parameter path:

```text
info["height_commands"] or info["commands_height"]
  -> RewardContext target height
  -> height reward helper
```

Test class: core param path.

Command:

```bash
uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k reward_context
```

Expected result:

- PASS for scalar fallback: old tasks still use `cfg.reward_config.base_height_target`.
- PASS for per-env target: two env rows with different targets produce row-specific rewards.

Probe/assert facts:

```text
old_context_target is scalar fallback
new_context_target.shape == (N,)
row0 target affects only row0 reward
row1 target affects only row1 reward
```

Stop condition:

- Height target can drive reward per env without touching actor/critic obs dims.

Evidence (2026-07-08):

- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k reward_context`: first failed because per-env target stayed as scalar fallback, then PASS (`2 passed, 6 deselected`) after connector patch.
- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q`: PASS (`8 passed`).
- `uv run python -m py_compile src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`: PASS.
- `uv run ruff check src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`: PASS.

### Step 1.5: SAC G1 Height Task Config Without Old Checkpoint Drift

Scope: create or identify a new owner config/task path for height-conditioned G1 so old `G1WalkFlat` checkpoint dims remain stable.

Non-scope: do not mutate existing `g1_walk_flat` obs dims in-place.

Files:

- Create: `conf/offpolicy/task/sac/g1_walk_height/mujoco.yaml`
  - Start by copying only the necessary owner fields from `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml`.
  - Set `training.task_name` to a new name such as `G1WalkHeight`.
  - Keep `training.sim_backend: mujoco`.
  - Add height command fields and height reward fields only in this new task.
- Create: `conf/offpolicy/task/sac/g1_walk_height/motrix.yaml` only after the MuJoCo contract passes, because backend parity is a second boundary.
- Do not modify: `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml` except if a shared base file is introduced in a separate refactor step.
- Test: `tests/config/test_reward_injection.py` or a new config contract test if the existing file is not the right owner.

Owner module: offpolicy SAC task config owner.

Core parameter path:

```text
task=sac/g1_walk_height/mujoco
  -> env.commands height fields
  -> env.reward_config height scale/std
  -> obs_groups_spec expected dims
```

Test class: secondary contract path.

Command:

```bash
uv run pytest tests/config/test_reward_injection.py -q -k "g1 and height"
```

Expected result:

- Existing `g1_walk_flat` SAC config still composes with old `obs_groups_spec`.
- New height task config composes with explicit height fields.
- Sim2Sim denylist fields are not silently changed for old task names.

Probe/assert facts:

```text
old_sac_g1_walk_flat obs dim unchanged
new_sac_g1_walk_height has height range/default
height reward scale exists only in new task or behind explicit flag
```

Stop condition:

- There is a named SAC/MuJoCo config boundary for height tracking, and old checkpoint compatibility is protected.

Evidence (2026-07-08):

- `uv run pytest tests/config/test_reward_injection.py -q -k "g1 and height"`: first failed with missing `task/sac/g1_walk_height/mujoco`, then PASS (`2 passed, 9 deselected`) after adding the new MuJoCo task config.
- `uv run pytest tests/config/test_reward_injection.py -q`: PASS (`11 passed, 2 warnings` from existing Gymnasium Box cast checks).
- `uv run python -m py_compile tests/config/test_reward_injection.py`: PASS.
- `uv run ruff check tests/config/test_reward_injection.py`: PASS.
- Scope note: this step declares the config boundary only; G1 reward dispatch for `track_base_height_exp_smooth` is not wired in this step.

### Step 1.6: Optional Observation Contract For Height Policy

Scope: decide and implement how target height reaches policy/critic observation for the new height task.

Non-scope: no distillation, no MoE, no old task obs dim mutation.

Files:

- Modify: `src/unilab/envs/locomotion/g1/joystick.py`
  - Add height command to actor/critic parts only when the new task/config enables it.
  - Update `obs_groups_spec` to compute dimensions from the enabled command layout, or add a separate env/task class if cleaner.
- Test: `tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`
  - Add obs dimension and ordering tests.

Owner module: `g1/joystick.py`.

Core parameter path:

```text
info["commands"] shape (N, 3)
optional height_cmd shape (N, 1)
  -> actor command block
  -> critic command block
  -> obs_groups_spec
```

Test class: shape contract.

Command:

```bash
uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k height_obs
```

Expected result:

- Old path: actor dim remains 98 or 99 depending on `mode_observation`.
- New height path: actor/critic dims increase by exactly 1 if height is appended.
- The appended value is the target height, not measured height.

Probe/assert facts:

```text
old_obs_dim unchanged
height_obs_dim == old_obs_dim + 1
critic_height_obs_dim == old_critic_dim + 1
height command column matches target height fixture
```

Stop condition:

- Observation shape and semantic column ordering are contract-confirmed.

Evidence (2026-07-08):

- Added explicit `env.commands.observe_height_command` so old `G1WalkFlat` checkpoint paths keep their original actor/critic dims while the new height task opts into a policy I/O change.
- `G1WalkEnv.obs_groups_spec`, actor obs, critic obs, and symmetry obs layout now use a 4-D command block only when `observe_height_command=true`; otherwise the command block stays 3-D.
- The height column is appended after `[vx, vy, yaw]` and comes from `info["height_commands"]` / `info["commands_height"]` / configured default target, not measured base height.
- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q -k height_obs`: first failed (`obs=99, critic=102` instead of `100/103`), then PASS (`2 passed, 8 deselected`).
- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q`: PASS (`10 passed`).
- `uv run pytest tests/config/test_reward_injection.py -q -k "g1 and height"`: PASS (`2 passed, 9 deselected`).
- `uv run pytest tests/config/test_reward_injection.py -q`: PASS (`11 passed, 2 warnings` from existing Gymnasium Box cast checks).
- `uv run python -m py_compile src/unilab/envs/locomotion/common/commands.py src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py tests/config/test_reward_injection.py`: PASS.
- `uv run ruff check src/unilab/envs/locomotion/common/commands.py src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py tests/config/test_reward_injection.py`: PASS.

### Step 1.7: One-Step Live Sentinel

Scope: run the cheapest real env path that crosses reset, command sampling, obs construction, reward construction, and logging.

Non-scope: no training, no visual quality claim, no long rollout.

Files:

- Create: `scripts/deploy/check_unilab_g1_height_tracking_live_path.py`
  - Responsibility: one-step or few-step sentinel for the new height task.
- Optional test wrapper: `tests/scripts/test_train_scripts.py` only if script invocation contract belongs there.

Owner module: live-path sentinel script.

Core parameter path:

```text
Hydra config
  -> env reset info
  -> sampled velocity and height command
  -> measured base height
  -> obs dims
  -> height reward log
```

Test class: live sentinel path.

Command:

```bash
uv run scripts/deploy/check_unilab_g1_height_tracking_live_path.py --num-envs 4 --steps 1
```

Expected result:

- Exit code 0.
- Prints one compact block with height target, measured height, reward term, old/new obs dims, and finite checks.
- Does not claim that the learned policy can track height.

Probe facts:

```text
height_tracking/config_task
height_tracking/commands_shape
height_tracking/target_height_min_max_mean
height_tracking/measured_height_min_max_mean
height_tracking/reward_mean
height_tracking/obs_dim
height_tracking/critic_dim
height_tracking/finite=True
```

Stop condition:

- The real reset -> step path reaches height command, measured height, obs, reward, and log without dimension drift.

Evidence (2026-07-08):

- Created `scripts/deploy/check_unilab_g1_height_tracking_live_path.py` as the one-step live sentinel.
- Registered `G1WalkHeight` on MuJoCo so the new height task has its own explicit env identity while reusing `G1WalkEnv`.
- Wired reset sampling to emit `info["height_commands"]` when `observe_height_command` or `random_height_during_walking` is enabled.
- Wired `track_base_height_exp_smooth` into `G1WalkEnv._reward_fns`.
- First live sentinel after script creation failed at `height_tracking/reward_log`: reward scale existed, but the mode reward term list did not include `track_base_height_exp_smooth`.
- Added `track_base_height_exp_smooth` to the new height task `reward.mode.balance_common_terms`; old `G1WalkFlat` config remains unchanged.
- `uv run scripts/deploy/check_unilab_g1_height_tracking_live_path.py --num-envs 4 --steps 1`: PASS after approval to run outside sandbox because sandboxed plain `uv run` could not access `~/.cache/uv`.
- Final live facts: `config_task=G1WalkHeight`, `commands_shape=[4, 3]`, `height_commands_shape=[4, 1]`, `obs_dim=100`, `critic_dim=103`, `finite=True`, `reward/track_base_height_exp_smooth` logged.
- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q`: PASS (`10 passed`).
- `uv run pytest tests/config/test_reward_injection.py -q`: PASS (`11 passed, 2 warnings` from existing Gymnasium Box cast checks).
- `uv run ruff check scripts/deploy/check_unilab_g1_height_tracking_live_path.py src/unilab/envs/locomotion/g1/joystick.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`: PASS.
- `uv run python -m py_compile scripts/deploy/check_unilab_g1_height_tracking_live_path.py src/unilab/envs/locomotion/g1/joystick.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py`: PASS.

### Phase 1 Completion Gate

Phase 1 is complete only when all of these are true:

- Step 1.1 command helper contract passes.
- Step 1.2 reward formula contract passes.
- Step 1.3 measured-height boundary passes.
- Step 1.4 `RewardContext` bridge passes scalar fallback and per-env target cases.
- Step 1.5 config compose confirms old task compatibility and new task ownership.
- Step 1.6 observation contract passes if height is exposed to policy/critic.
- Step 1.7 live sentinel exits 0 and prints the expected structured facts.

Status: COMPLETE (2026-07-08).

Do not start Phase 2 distillation before this completion gate is satisfied.

## Phase 2: Migrate Agile G1 Distillation

### Execution Contract

Scope: add a G1 teacher-student distillation path in UniLab after Phase 1 height contracts are stable.

Non-scope: no MoE router yet, no multi-teacher target matrix, no reward changes.

Owner module: algorithm/training layer, not env scripts.

Core parameter path:

```text
teacher checkpoint path
  -> teacher policy load and input dimension guard
  -> teacher/privileged observation group
  -> teacher action target
  -> distillation storage field
  -> student action prediction
  -> MSE/Huber behavior loss
  -> checkpoint save/load
  -> playback with student-only observations
```

Agile source behavior:

```text
Distillation.act(obs, teacher_obs)
  -> student action executes in env
  -> teacher TorchScript action is stored as privileged_actions
  -> update() minimizes loss(student_action, privileged_actions)
```

Planned files:

- Create: `src/unilab/algos/torch/distill/`
  - Responsibility: generic behavior-cloning distillation components for locomotion.
- Create or modify: `scripts/train_distill.py` or a config-driven train entry.
  - Responsibility: assemble env, teacher policy, student policy, storage, and trainer.
- Modify: `src/unilab/visualization/`
  - Responsibility: playback student checkpoint with student obs only.
- Create: `conf/distill/task/g1_walk_height/*`
  - Responsibility: owner configs for G1 teacher-student distillation.
- Create: `tests/algos/test_g1_distillation_contract.py`
  - Responsibility: fake teacher and fake batch prove loss path and checkpoint contract.

Test class: core param path.

Probe facts:

```text
teacher_checkpoint path
teacher_obs shape
student_obs shape
teacher_action shape
student_action shape
loss value and requires_grad
student grad norm
optimizer update count
checkpoint keys
playback obs source
```

Expected result:

- Student loss has gradient only through student parameters.
- Teacher outputs are detached.
- Playback does not require privileged teacher observations.

Stop condition:

- Fake-batch contract passes and a one-update sentinel prints nonzero distillation update count and finite behavior loss.

### Step 2.1: Generic Behavior Distillation Offline Contract

Scope: establish the UniLab-native offline teacher-student behavior distillation
contract before adding Hydra config, live env rollout, playback, or MoE.

Non-scope:

- no `conf/distill` Hydra root yet;
- no `scripts/train_distill.py` live entrypoint yet;
- no MoE router or multi-teacher target matrix;
- no SAC/offpolicy runner changes.

Files:

- Created: `src/unilab/algos/torch/distill/`
  - `models.py`: `MLPStudentPolicy`, the deployable student actor.
  - `trainer.py`: `BehaviorDistillationTrainer`, `DistillationBatch`, and update stats.
  - `checkpoint.py`: student checkpoint save/load helpers.
- Created: `tests/algos/test_g1_distillation_contract.py`
  - Fake-batch tests for teacher detach, student gradient/update, checkpoint roundtrip, and shape mismatch.

Owner module: `src/unilab/algos/torch/distill/`.

Core parameter path:

```text
student_obs, teacher_obs
  -> teacher_action under torch.no_grad()
  -> detached teacher_action target
  -> student_action
  -> behavior loss
  -> student-only gradient and optimizer update
  -> student_state_dict checkpoint
```

Test class: core param path.

Command:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q
```

Expected result:

- Before implementation: FAIL with missing `unilab.algos.torch.distill`.
- After implementation: PASS, with nonzero student grad norm, detached teacher target, no teacher gradients, and checkpoint roundtrip.

Evidence (2026-07-09):

- First run failed as expected: `ModuleNotFoundError: No module named 'unilab.algos.torch.distill'`.
- After adding the generic distill package, `uv run pytest tests/algos/test_g1_distillation_contract.py -q`: PASS (`3 passed`).

Status: COMPLETE for offline core semantics only. Live env rollout, Hydra owner config,
teacher checkpoint loading from real SAC runs, playback, and MoE remain unconfirmed.

### Step 2.2: Distill Owner Config And SAC Teacher Load Contract

Scope: add the explicit distillation Hydra owner group and the offline SAC teacher
checkpoint loader used by future `train_distill.py`.

Non-scope:

- no live env rollout;
- no `scripts/train_distill.py` entrypoint assembly yet;
- no student playback adapter;
- no MoE.

Files:

- Created: `conf/distill/config.yaml`
  - Root owner for generic behavior distillation.
- Created: `conf/distill/task/g1_walk_height/mujoco.yaml`
  - G1 height distillation owner task.
- Created: `src/unilab/algos/torch/distill/teacher.py`
  - `DistillationTeacherSpec`, `LoadedTeacherPolicy`, and `load_sac_teacher_policy`.
- Modified: `tests/config/test_config_system.py`
  - Adds distill config compose and owner-parameter assertions.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds SAC teacher checkpoint load and dim-mismatch tests.

Owner flag / config group:

```text
flag name: config root `conf/distill` with `task=g1_walk_height/mujoco`
OFF behavior: existing `conf/offpolicy` and `conf/hora_distill` paths do not read distill config.
ON behavior: distill root resolves teacher SAC owner, teacher/student obs/action dims, and offline behavior loss defaults together.
generated/derived overrides: none yet; future entrypoint should read these fields without inventing Python defaults.
forbidden mixed states: teacher/student obs/action dims must not disagree silently; checkpoint dim mismatch must fail through `policy_load_dim_guard`.
```

Parameter inventory:

| param | owner | old default | new default | OFF value | ON value | consumers | persistence/playback risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `algo.algo_log_name` | `conf/distill/config.yaml` | absent | `distill` | absent | `distill` | future log root | low, new root only |
| `algo.gradient_length` | `conf/distill/config.yaml` | absent | `15` | absent | `15` | future trainer accumulation | no old checkpoint risk |
| `algo.loss_type` | `conf/distill/config.yaml` | absent | `mse` | absent | `mse` | `BehaviorDistillationTrainer` | no old checkpoint risk |
| `teacher.algo_family` / `teacher.algo_type` | `conf/distill/*` | absent | `sac` | absent | `sac` | teacher loader | real checkpoint load risk |
| `teacher.task` | `conf/distill/task/g1_walk_height/mujoco.yaml` | absent | `sac/g1_walk_height/mujoco` | absent | SAC height teacher owner | future checkpoint resolver | high if wrong run root |
| `teacher.obs_dim` | `conf/distill/*` | absent | `99` | absent | `99` by default, `100` only by explicit legacy override | `load_sac_teacher_policy` | high, checkpoint dim guard |
| `teacher.action_dim` | `conf/distill/*` | absent | `29` | absent | `29` | `load_sac_teacher_policy` | high, checkpoint dim guard |
| `student.obs_dim` | `conf/distill/*` | absent | `99` | absent | `99` | `MLPStudentPolicy`, student-only playback | high for playback; live actor obs is 99-D |
| `student.action_dim` | `conf/distill/*` | absent | `29` | absent | `29` | `MLPStudentPolicy` | high for playback |

Core parameter path:

```text
conf/distill task selection
  -> teacher spec
  -> SAC actor build
  -> checkpoint["actor"] load under policy_load_dim_guard
  -> frozen LoadedTeacherPolicy
  -> teacher action shape (N, 29)
```

Test class: secondary config contract plus checkpoint shape core param path.

Commands:

```bash
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/algos/test_g1_distillation_contract.py -q
```

Expected result:

- `conf/distill` composes with `task=g1_walk_height/mujoco`.
- SAC teacher checkpoint loads and emits finite detached action tensors.
- SAC teacher checkpoint obs/action dim mismatch is re-raised as `CrossBackendIncompatibleError`.

Evidence (2026-07-09):

- Before implementation, config test failed with missing `conf/distill`.
- Before implementation, teacher tests failed with missing `DistillationTeacherSpec`.
- After implementation, `uv run pytest tests/config/test_config_system.py -q -k distill`: PASS (`3 passed, 117 deselected`).
- After implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q`: PASS (`5 passed`).

Status: COMPLETE for offline config and SAC teacher checkpoint loading only.
Next boundary at this point: entrypoint assembly. Live rollout, student playback,
and MoE remain unconfirmed.

### Step 2.3: Distill Entrypoint Assembly And Fake-Batch Probe

Scope: add a UniLab-native `scripts/train_distill.py` assembly layer that reads
`conf/distill`, resolves SAC teacher checkpoints through shared training path
semantics, builds the frozen teacher/student/trainer objects, and runs one
offline fake-batch update probe.

Non-scope:

- no live env rollout or collector/storage data path;
- no formal distillation training loop;
- no student playback adapter;
- no MoE.

Files:

- Created: `scripts/train_distill.py`
  - `build_teacher_spec(cfg)`: maps Hydra owner fields to `DistillationTeacherSpec`.
  - `build_student_policy(cfg)`: builds `MLPStudentPolicy` from `cfg.student`.
  - `resolve_teacher_checkpoint(cfg, root_dir=...)`: delegates to `resolve_task_checkpoint_path`.
  - `build_distillation_trainer(...)`: loads the frozen SAC teacher and assembles `BehaviorDistillationTrainer`.
  - `run_fake_batch_update(...)`: executes one deterministic, shape-valid offline update.
- Modified: `conf/distill/config.yaml`
  - Adds explicit `training.dry_run: false` owner field for the offline probe mode.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds entrypoint assembly, checkpoint resolver, and fake-batch update tests.
- Modified: `tests/config/test_config_system.py`
  - Confirms the dry-run flag is config-owned.

Owner module: `scripts/train_distill.py` as entrypoint assembly only; long-lived
algorithm semantics remain in `src/unilab/algos/torch/distill`.

Core parameter path:

```text
conf/distill
 -> teacher spec
 -> resolve_task_checkpoint_path(root, task_name, load_run, algo_log_name, checkpoint)
 -> load_sac_teacher_policy(...)
 -> MLPStudentPolicy(...)
 -> BehaviorDistillationTrainer.update(fake DistillationBatch)
 -> finite loss, detached teacher action, student grad norm, update_count=1
```

Test class: entrypoint connectivity plus fake-batch core param path.

Commands:

```bash
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/algos/test_g1_distillation_contract.py -q
```

Expected result:

- `scripts/train_distill.py` imports without side effects.
- The teacher checkpoint resolver forwards `task_name=G1WalkHeight`, `algo_log_name=fast_sac`, `teacher.load_run`, `teacher.checkpoint`, and `training.log_root`.
- One fake-batch update reports teacher observations `(batch, 100)`, student observations `(batch, 99)`, `(batch, 29)` actions, detached teacher action, finite loss, nonzero student grad norm, and `update_count=1`.

Evidence (2026-07-09):

- Before implementation, the new script tests failed as expected with missing `scripts/train_distill.py`.
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k distill`: PASS (`19 passed, 133 deselected`).

Status: COMPLETE for entrypoint assembly and fake-batch probe only.
Next boundary at this point: offline dataset and checkpoint path. Live env rollout,
formal training loop, interactive/live playback session, and MoE remain unconfirmed.

### Step 2.4: Offline Distillation Dataset Boundary

Scope: add a UniLab-native offline dataset boundary for behavior distillation so
future live collector or offline replay code must pass through explicit
student/teacher observation validation before reaching the trainer.

Non-scope:

- no live env rollout or collector lifecycle;
- no storage queue, replay buffer, or async runner changes;
- no cached teacher action target format;
- no formal distillation training loop;
- no student playback adapter;
- no MoE.

Files:

- Created: `src/unilab/algos/torch/distill/data.py`
  - `DistillationTensorDataset`: in-memory student/teacher observation dataset.
  - `build_distillation_dataset(...)`: validates rank, batch size, obs dims, and finite values.
  - `make_fake_distillation_dataset(...)`: deterministic fake data for offline connectivity probes.
  - `save_distillation_dataset(...)` / `load_distillation_dataset(...)`: persistence roundtrip with dim guards.
- Modified: `src/unilab/algos/torch/distill/__init__.py`
  - Exports dataset helpers.
- Modified: `scripts/train_distill.py`
  - `run_fake_batch_update(...)` now routes through `make_fake_distillation_dataset(...).as_batch(...)`.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds dataset roundtrip, slicing, dim mismatch, batch-size mismatch, and finite-value checks.
- Modified: `tests/scripts/test_train_scripts.py`
  - Confirms entrypoint fake probe reports dataset boundary facts.

Owner module: `src/unilab/algos/torch/distill/data.py`.

Core parameter path:

```text
student_obs, teacher_obs
 -> build_distillation_dataset(...)
 -> rank/batch/dim/finite guards
 -> DistillationTensorDataset.as_batch(...)
 -> DistillationBatch
 -> BehaviorDistillationTrainer.update(...)
```

Test class: core param path plus persistence contract.

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k dataset
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
```

Expected result:

- Bad dataset shapes fail before trainer update.
- Dataset save/load preserves observations, metadata, and dimensions.
- `train_distill.py` fake probe reports dataset sample count and obs dims before loss/update facts.

Evidence (2026-07-09):

- Before implementation, dataset tests failed with missing `build_distillation_dataset`.
- Before implementation, script probe test failed with missing `dataset_num_samples`.
- After implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q -k dataset`: PASS (`2 passed, 5 deselected`).
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k distill`: PASS (`19 passed, 133 deselected`).

Status: COMPLETE for offline dataset boundary only.
Live collector, replay/storage integration, cached teacher-action targets,
formal training loop, interactive/live playback session, and MoE remain unconfirmed.

### Step 2.5: Offline Micro-Training Loop And Student Checkpoint

Scope: add a bounded offline update loop that consumes a validated
`DistillationTensorDataset`, updates the student for a small number of batches,
and optionally writes a student-only distillation checkpoint.

Non-scope:

- no live env rollout or collector lifecycle;
- no replay/storage queue integration;
- no cached teacher-action target format;
- no formal long-running training loop;
- no student playback adapter;
- no MoE.

Files:

- Created: `src/unilab/algos/torch/distill/offline.py`
  - `OfflineDistillationRunResult`: summary of bounded offline updates.
  - `run_offline_distillation_updates(...)`: sequential dataset batches -> trainer updates -> optional student checkpoint.
- Modified: `src/unilab/algos/torch/distill/__init__.py`
  - Exports offline loop result and runner.
- Modified: `scripts/train_distill.py`
  - `run_fake_batch_update(...)` now supports `max_updates` and optional `checkpoint_path`.
  - `training.dry_run=true` routes through the bounded offline loop, not a one-off direct trainer call.
- Modified: `conf/distill/config.yaml`
  - Adds `training.dry_run_batch_size`, `training.dry_run_updates`, and `training.dry_run_checkpoint`.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds offline update loop and checkpoint roundtrip contract.
- Modified: `tests/scripts/test_train_scripts.py`
  - Confirms entrypoint fake probe can run multiple updates and save a student checkpoint.
- Modified: `tests/config/test_config_system.py`
  - Confirms dry-run loop parameters are config-owned.

Owner module: `src/unilab/algos/torch/distill/offline.py`.

Core parameter path:

```text
DistillationTensorDataset
 -> as_batch(start, batch_size)
 -> BehaviorDistillationTrainer.update(...)
 -> OfflineDistillationRunResult(update_count, samples_seen, loss, grad norm)
 -> save_distillation_checkpoint(student_state_dict, optimizer_state_dict, agent_steps)
 -> load_distillation_checkpoint(...)
```

Feature flag contract:

```text
flag name: training.dry_run under conf/distill
OFF behavior: no offline update loop runs from train_distill main.
ON behavior: train_distill resolves teacher checkpoint, builds fake dataset, runs bounded offline updates, and optionally writes a student checkpoint.
generated/derived overrides: dry_run dataset samples = dry_run_batch_size * dry_run_updates.
forbidden mixed states: batch_size <= 0, max_updates <= 0, or empty dataset fail before update.
tests proving OFF: config compose confirms dry_run=false and loop params inert by default.
tests proving ON: script fake probe performs two updates, sees four samples, writes loadable student checkpoint.
```

Test class: core param path plus persistence contract.

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k offline_distillation_run
uv run pytest tests/scripts/test_train_scripts.py -q -k distill_script_fake_batch_probe
uv run pytest tests/config/test_config_system.py -q -k distill_g1
```

Expected result:

- Offline loop increments update count and sample count.
- Saved checkpoint contains student state, optimizer state, `agent_steps`, teacher metadata, and runtime config.
- Entry probe returns dataset facts, loss/grad facts, update count, sample count, and checkpoint path.

Evidence (2026-07-09):

- Before implementation, offline loop test failed with missing `run_offline_distillation_updates`.
- Before implementation, script probe failed because `run_fake_batch_update` did not accept `max_updates`.
- Before implementation, config test failed because `training.dry_run_batch_size` was absent.
- After implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q -k offline_distillation_run`: PASS (`1 passed, 7 deselected`).
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k distill_script_fake_batch_probe`: PASS (`1 passed, 151 deselected`).
- After implementation, `uv run pytest tests/config/test_config_system.py -q -k distill_g1`: PASS (`1 passed, 119 deselected`).

Status: COMPLETE for bounded offline micro-training and student checkpoint only.
Live collector, replay/storage integration, cached teacher-action targets,
formal long-running training, live viewer/MuJoCo rollout, and MoE remain unconfirmed.

### Step 2.6: Student-Only Playback Load Contract

Scope: add the student-only checkpoint load boundary required before wiring a
generic distillation student into interactive playback.

Non-scope:

- no viewer or interactive playback session construction;
- no live env reset/step;
- no replay/storage queue integration;
- no formal long-running training;
- no MoE.

Files:

- Created: `src/unilab/algos/torch/distill/playback.py`
  - `LoadedDistillationStudentPolicy`: frozen student policy plus checkpoint metadata.
  - `load_distillation_student_policy(...)`: checkpoint -> `MLPStudentPolicy` -> student-only callable.
- Modified: `src/unilab/algos/torch/distill/__init__.py`
  - Exports student playback helper.
- Modified: `scripts/train_distill.py`
  - Dry-run checkpoint runtime cfg now stores student obs/action dims, hidden dims, activation, and squash flag.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds student-only checkpoint load, action shape, no-grad action, and missing-runtime-dim rejection.
- Modified: `tests/scripts/test_train_scripts.py`
  - Confirms dry-run checkpoint has enough student architecture metadata for playback helper loading.

Owner module: `src/unilab/algos/torch/distill/playback.py`.

Core parameter path:

```text
distillation checkpoint
 -> distill_runtime_cfg.student_obs_dim/student_action_dim/student_hidden_dims
 -> MLPStudentPolicy(...)
 -> load_distillation_checkpoint(...)
 -> policy(student_obs)
 -> action shape (N, action_dim), no teacher/privileged obs required
```

Test class: checkpoint/playback compatibility contract.

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "student_checkpoint_loads_for_student_only_playback or student_playback_rejects"
uv run pytest tests/scripts/test_train_scripts.py -q -k distill_script_fake_batch_probe
```

Expected result:

- Student checkpoint loads without teacher policy or teacher observations.
- Loaded student rejects wrong actor obs dim explicitly.
- Entry dry-run checkpoint stores enough runtime metadata to reconstruct the student.

Evidence (2026-07-09):

- Before implementation, student playback test failed with missing `load_distillation_student_policy`.
- After implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "student_checkpoint_loads_for_student_only_playback or student_playback_rejects"`: PASS (`2 passed, 8 deselected`).
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k distill_script_fake_batch_probe`: PASS (`1 passed, 151 deselected`).

Status: COMPLETE for student-only checkpoint load and action I/O only.
Live collector, replay/storage integration, cached teacher-action targets,
formal long-running training, live viewer rollout, and MoE remain unconfirmed.

### Step 2.7: Generic Distill Interactive Playback Session Adapter

Scope: wire generic distillation student checkpoints into the existing
`RslRlPlaybackSession` contract with a fake-env connectivity test.

Non-scope:

- no real MuJoCo env reset/step;
- no viewer launch;
- no CLI routing in `scripts/play_interactive.py`;
- no replay/storage queue integration;
- no formal long-running training;
- no MoE.

Files:

- Modified: `src/unilab/visualization/interactive_playback.py`
  - Adds `create_distill_playback_session(...)`.
  - Adds `_distill_student_obs_tensor(...)` for tensor/numpy/dict-like actor obs.
  - Reuses `RslRlPlaybackSession` and falls back to zero actions when no checkpoint is found.
- Modified: `tests/visualization/test_interactive_playback.py`
  - Adds fake-env tests for checkpoint-loaded student actions and missing-checkpoint zero actions.

Owner module: `src/unilab/visualization/interactive_playback.py` as playback session adapter; student checkpoint semantics remain in `src/unilab/algos/torch/distill`.

Core parameter path:

```text
RslRlPlaybackConfig(action_mode=policy)
 -> resolve generic distill student checkpoint
 -> load_distillation_student_policy(...)
 -> wrapper.reset() actor obs
 -> _distill_student_obs_tensor(...)
 -> student.policy(obs)
 -> RslRlPlaybackSession.step_once() actions
```

Feature flag contract:

```text
flag name: algo_log_name=distill plus explicit create_distill_playback_session call
OFF behavior: existing HORA/SAC playback factories do not import or call generic distill adapter.
ON behavior: generic distill session loads a student checkpoint and uses only student actor obs for actions.
generated/derived overrides: policy_obs_mode=auto normalizes to actor.
forbidden mixed states: missing checkpoint logs warning and uses zero actions; wrong obs dim fails in student policy.
tests proving OFF: HORA/SAC playback test subset still passes.
tests proving ON: generic distill fake-env session produces finite student actions.
```

Test class: offline playback connectivity contract.

Commands:

```bash
uv run pytest tests/visualization/test_interactive_playback.py -q -k "create_distill_playback_session"
uv run pytest tests/visualization/test_interactive_playback.py -q -k "hora_distill or sac_playback or create_distill_playback_session"
```

Expected result:

- Existing HORA/SAC playback behavior remains unchanged.
- Generic distill session loads a student checkpoint and steps once with finite `(num_envs, action_dim)` actions.
- Missing checkpoint path falls back to zero actions with an explicit warning.

Evidence (2026-07-10):

- Before implementation, test collection failed because `create_distill_playback_session` was not exported.
- After implementation, `uv run pytest tests/visualization/test_interactive_playback.py -q -k "create_distill_playback_session"`: PASS (`2 passed, 15 deselected`).
- After implementation, `uv run pytest tests/visualization/test_interactive_playback.py -q -k "hora_distill or sac_playback or create_distill_playback_session"`: PASS (`8 passed, 9 deselected`).

Status: COMPLETE for fake-env interactive playback session adapter only.
Live viewer rollout, live MuJoCo reset/step, replay/storage integration,
formal long-running training, and MoE remain unconfirmed.

### Step 2.8: Generic Distill Interactive CLI Routing Contract

Scope: route `scripts/play_interactive.py --algo distill` into the generic
distillation playback session factory and add `interactive` defaults to the
distill Hydra owner config.

Non-scope:

- no viewer launch;
- no real MuJoCo env reset/step;
- no policy-quality validation;
- no replay/storage integration;
- no formal long-running training;
- no MoE.

Files:

- Modified: `scripts/play_interactive.py`
  - Adds `distill` to `SUPPORTED_INTERACTIVE_ALGOS`.
  - Maps `distill` to `conf/distill`.
  - Routes `algo == "distill"` to `create_distill_playback_session(...)`.
- Modified: `conf/distill/config.yaml`
  - Adds owner `interactive` defaults matching existing playback configs.
- Modified: `tests/scripts/test_visualization_entrypoints.py`
  - Adds CLI compose and session routing contract tests.
- Modified: `tests/config/test_config_system.py`
  - Guards distill `interactive.action_mode` and `policy_obs_mode` defaults.

Owner module: `scripts/play_interactive.py` owns interactive CLI/session
routing; `conf/distill/config.yaml` owns distill playback defaults.

Core parameter path:

```text
--algo distill --task g1_walk_height --sim mujoco
 -> _parse_interactive_cli(...)
 -> _compose_interactive_config("distill", ["task=g1_walk_height/mujoco", ...])
 -> _build_play_args(...)
 -> _build_playback_config(...)
 -> create_distill_playback_session(...)
```

Feature flag contract:

```text
flag name: --algo distill
OFF behavior: ppo/appo/sac/flashsac/hora_distill routing is unchanged.
ON behavior: distill config composes from conf/distill and uses generic distill session.
generated/derived overrides: --task/--sim still generate task=<task>/<sim>.
forbidden mixed states: --task cannot contain '/', --sim cannot contain '/', task/training.sim_backend overrides remain rejected by the shared parser.
tests proving OFF: visualization entrypoint and touched playback subsets still pass.
tests proving ON: distill CLI compose test plus monkeypatched session factory route test.
```

Test class: secondary contract path plus offline session-routing connectivity.

Commands:

```bash
uv run pytest tests/scripts/test_visualization_entrypoints.py -q -k distill
uv run pytest tests/config/test_config_system.py -q -k distill
```

Expected result:

- `--algo distill` is accepted by the parser.
- distill Hydra config composes and exposes `interactive` defaults.
- `play_interactive(..., algo="distill")` calls `create_distill_playback_session(...)`.
- The test exits before viewer launch by monkeypatching GLFW availability, so it is not a live viewer claim.

Evidence (2026-07-10):

- Before implementation, `uv run pytest tests/scripts/test_visualization_entrypoints.py -q -k distill`: FAIL because argparse rejected `distill` and `create_distill_playback_session` was not imported in the entrypoint.
- After implementation, `uv run pytest tests/scripts/test_visualization_entrypoints.py -q -k distill`: PASS (`2 passed, 15 deselected`).

Status: COMPLETE for interactive CLI/session-factory routing only.
Live viewer rollout, replay/storage integration, formal long-running training,
policy-checkpoint live rollout, and MoE remained unconfirmed at this step.

### Step 2.9: Generic Distill Live MuJoCo Playback Sentinel

Scope: add a minimal live MuJoCo reset/step sentinel for generic distillation
playback and close the owner-config gap exposed by live env construction.

Non-scope:

- no viewer window launch;
- no policy-quality validation;
- no real student checkpoint rollout;
- no replay/storage integration;
- no formal long-running training;
- no MoE.

Files:

- Created: `scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py`
  - Composes `conf/distill` and builds `create_distill_playback_session(...)`.
  - Runs one real MuJoCo reset/step in `action_mode=zero`.
  - Prints compact runtime facts for policy obs mode, action dim, physics shape, action shape, and info keys.
- Modified: `conf/distill/task/g1_walk_height/mujoco.yaml`
  - Adds the complete G1 height env/reward owner contract needed by live env construction.
- Modified: `src/unilab/visualization/interactive_playback.py`
  - Generic distill playback now uses `BackendAdapter(...).build_task_env_cfg_override()`.
  - `RslRlPlaybackSession` stores last actions so live sentinels can inspect action shape/device.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds fake-session contract coverage for the live sentinel script.
- Modified: `tests/config/test_config_system.py`
  - Guards that distill G1 height config includes height-command and height-reward owner fields.

Owner module: `src/unilab/visualization/interactive_playback.py` owns playback
session construction; `conf/distill/task/g1_walk_height/mujoco.yaml` owns the
live G1 height env/reward config used by generic distill playback.

Core parameter path:

```text
conf/distill task=g1_walk_height/mujoco
 -> BackendAdapter.build_task_env_cfg_override()
 -> create_env(..., task_name=G1WalkHeight, sim_backend=mujoco)
 -> create_distill_playback_session(...)
 -> RslRlPlaybackSession.reset()
 -> RslRlPlaybackSession.step_once()
 -> actions / physics_state / info runtime facts
```

Feature flag contract:

```text
flag name: --algo distill plus conf/distill task owner
OFF behavior: existing PPO/APPO/SAC/HORA playback factories and configs are unchanged.
ON behavior: generic distill playback creates the G1WalkHeight MuJoCo env through distill owner config.
generated/derived overrides: env/reward owner fields are read from conf/distill/task/g1_walk_height/mujoco.yaml via BackendAdapter.
forbidden mixed states: missing reward/env owner fields fail before live sentinel can pass.
tests proving OFF: HORA/SAC/generic playback subset still passes.
tests proving ON: live sentinel reaches reset/step and reports finite physics/actions.
```

Test class: live sentinel path for env construction/reset/step plus secondary
contract checks for config ownership.

Commands:

```bash
uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --steps 1 --action-mode zero --device cpu
uv run pytest tests/scripts/test_train_scripts.py -q -k distill_playback_live_sentinel
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/visualization/test_interactive_playback.py -q -k "hora_distill or sac_playback or create_distill_playback_session"
```

Expected result:

- Live MuJoCo env creation succeeds for `G1WalkHeight`.
- `policy_obs_mode=actor`.
- `action_dim=29`.
- `physics_state` and `actions` are finite with expected batch/action shape.
- The test uses `action_mode=zero`; it proves lifecycle wiring, not policy quality.

Evidence (2026-07-10):

- Before owner-config fix, live sentinel failed with `Environment 'G1WalkHeight' is not registered`, then `reward_config must be provided via Hydra configuration`, then `Missing 'reward' config in Hydra`.
- After implementation, `uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --steps 1 --action-mode zero --device cpu`: PASS.
- Runtime facts: `policy_obs_mode=actor`, `action_dim=29`, `physics_shape=[1, 72]`, `actions_shape=[1, 29]`, info keys include `commands`, `height_commands`, `current_actions`, `executed_actions`, `log`, and `timing`.
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k distill_playback_live_sentinel`: PASS (`1 passed, 152 deselected`).
- After implementation, `uv run pytest tests/config/test_config_system.py -q -k distill`: PASS (`3 passed, 117 deselected`).

Status: COMPLETE for live MuJoCo zero-action reset/step sentinel.
Viewer window launch, trained student-checkpoint policy rollout, replay/storage integration,
formal long-running training, and MoE remain unconfirmed.

### Step 2.10: Generic Distill Policy-Mode Live Sentinel

Scope: extend the generic distillation live sentinel from zero-action lifecycle
coverage to a policy-mode path with a temporary deployable student checkpoint.

Non-scope:

- no viewer window launch;
- no trained policy quality claim;
- no replay/storage integration;
- no formal long-running training;
- no MoE.

Files:

- Modified: `scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py`
  - Adds `--make-temp-policy-checkpoint`.
  - Creates a temporary `model_1.pt` with `student.obs_dim`, `student.action_dim`,
    hidden dims, activation, and squash metadata.
  - Requires `action_mode=policy` for the temp checkpoint probe.
  - Prints checkpoint path and action nonzero facts.
- Modified: `conf/distill/config.yaml`
  - Historical note: this step originally kept `teacher.obs_dim=100` for the
    SAC teacher checkpoint route. Step 3.8 supersedes that default with 99-D
    live-env identity and leaves 100-D as an explicit legacy override.
  - Sets `student.obs_dim=99`, matching the live `G1WalkHeight` actor obs used
    by student-only playback.
- Modified: `conf/distill/task/g1_walk_height/mujoco.yaml`
  - Historical note: this step originally kept the same 100/99 teacher/student
    split. Step 3.8 supersedes it with the 99/99 default.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds fake-session contract coverage for policy-mode temp-checkpoint loading.
  - Updates fake-batch checkpoint restore assertions to `student_obs_dim=99`.
- Modified: `tests/config/test_config_system.py`
  - Historical note: this step originally guarded `teacher.obs_dim == 100` and
    `student.obs_dim == 99`; Step 3.8 updates the current guard to 99/99.

Owner module: `scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py`
owns the live sentinel; `conf/distill` owns the teacher/student dimension split.

Core parameter path:

```text
conf/distill task=g1_walk_height/mujoco
 -> temporary distillation student checkpoint with student_obs_dim=99
 -> create_distill_playback_session(...)
 -> load_distillation_student_policy(...)
 -> wrapper.reset() actor obs
 -> student.policy(actor_obs)
 -> nonzero 29-D action
 -> RslRlPlaybackSession.step_once()
 -> MuJoCo physics state / action runtime facts
```

Feature flag contract:

```text
flag name: --make-temp-policy-checkpoint with --action-mode policy
OFF behavior: zero-action live sentinel and missing-checkpoint fallback remain unchanged.
ON behavior: sentinel uses a temporary student checkpoint and proves policy action flow.
generated/derived overrides: load_run points to a TemporaryDirectory containing model_1.pt.
forbidden mixed states: temp policy checkpoint is rejected unless action_mode=policy.
tests proving OFF: existing zero-action sentinel contract remains covered.
tests proving ON: fake policy checkpoint contract plus one-step live MuJoCo policy sentinel.
```

Test class: live sentinel path plus checkpoint/playback compatibility contract.

Commands:

```bash
uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --steps 1 --action-mode policy --make-temp-policy-checkpoint --device cpu
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/visualization/test_interactive_playback.py -q -k "hora_distill or sac_playback or create_distill_playback_session"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
```

Expected result:

- Student checkpoint is resolved and loaded.
- `policy_obs_mode=actor`.
- `action_dim=29`.
- `actions_abs_max > 0`, proving policy-mode action is not the zero fallback.
- `physics_state` and `actions` are finite with expected shape.

Evidence (2026-07-10):

- First policy-mode live run exposed the real dimension split: teacher checkpoint
  route is 100-D, but live student actor obs is 99-D (`Student obs dim mismatch:
  expected 100, got 99`).
- After fixing the owner config split, `uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --steps 1 --action-mode policy --make-temp-policy-checkpoint --device cpu`: PASS.
- Runtime facts: `policy_obs_mode=actor`, `checkpoint_path=<temp>/model_1.pt`,
  `action_dim=29`, `physics_shape=[1, 72]`, `actions_shape=[1, 29]`,
  `actions_abs_max=0.049958`, and info keys include `commands`,
  `height_commands`, `current_actions`, `executed_actions`, `log`, and `timing`.
- `uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"`: PASS (`21 passed, 133 deselected`).
- `uv run pytest tests/config/test_config_system.py -q -k distill`: PASS (`3 passed, 117 deselected`).
- `uv run pytest tests/visualization/test_interactive_playback.py -q -k "hora_distill or sac_playback or create_distill_playback_session"`: PASS (`8 passed, 9 deselected`).
- `uv run pytest tests/algos/test_g1_distillation_contract.py -q`: PASS (`10 passed`).

Status: COMPLETE for temp student-checkpoint policy-mode live MuJoCo reset/step.
Viewer window launch, trained student-checkpoint policy quality,
replay/storage integration, formal long-running training, and MoE remain unconfirmed.

## Phase 3: Add MoE Student Distillation

### Execution Contract

Scope: replace or extend the student network with a small MoE student while preserving the Phase 2 distillation API.

Non-scope: no optimized distributed MoE, no token-dropping transformer MoE dependency, no GPL code copy.

Owner module: student policy module plus distillation loss diagnostics.

Core parameter path:

```text
student_obs
  -> router logits
  -> route probabilities or hard expert id
  -> expert action outputs
  -> mixed action
  -> behavior loss against teacher action
  -> router/load-balance diagnostics
  -> checkpoint/export/playback
```

Recommended first implementation:

```text
MoEStudent(
  router_input = command + projected_gravity + joint state summary,
  experts = [standing_or_low_speed, walking, recovery],
  routing = semi-hard by command/mode first, learned soft gate second,
  output = action_dim,
)
```

Why semi-hard first:

- Standing/walking/recovery are already semantically visible from command and disturbance state.
- A fully unsupervised router can collapse to one expert under single-teacher MSE.
- Explicit routing gives interpretable diagnostics before expensive training.

External code policy:

- `lucidrains/mixture-of-experts` can be used as a permissive reference for top-2 gating, capacity factor, and aux loss.
- `davidmrau/mixture-of-experts` is GPL-3, so only use it to understand concepts such as sparse dispatch; do not copy implementation.
- For first robotics prototype, prefer a simple local PyTorch `ModuleList` MoE over adding a dependency.

Planned files:

- Create: `src/unilab/algos/torch/distill/moe_student.py`
  - Responsibility: small action-space MoE student, router diagnostics, optional hard routing.
- Future modify: `src/unilab/algos/torch/distill/trainer.py`
  - Responsibility: add `aux_loss` and router diagnostics to behavior loss.
- Future modify: `conf/distill/task/g1_walk_height/*`
  - Responsibility: expose `num_experts`, `routing_mode`, `aux_loss_coef`, and expert role labels.
- Test: `tests/algos/test_g1_distillation_contract.py`
  - Responsibility: prove routing shape, expert usage count, aux loss finite, and action shape.

Test class: core param path plus shape contract.

Probe facts:

```text
router_logits shape
route_probs finite/min/max
expert_usage counts
expert_action shape
mixed_action shape
behavior_loss
aux_loss
total_loss requires_grad
router grad norm
expert grad norms
```

Expected result:

- Every batch reports expert usage.
- Behavior loss remains finite.
- Router and at least one expert receive gradients in soft mode.
- Hard mode produces deterministic expert ids from command/mode labels.

Stop condition:

- Fake-batch MoE contract passes and one live distillation update prints expert usage for standing, walking, and recovery samples.

### Step 3.1: MoE Student Offline Routing Contract

Scope: add the smallest UniLab-native action-space MoE student module and prove
its local routing, expert mixing, and gradient paths offline.

Non-scope:

- no trainer integration;
- no aux loss in `BehaviorDistillationTrainer`;
- no Hydra `student.model_type=moe` wiring;
- no checkpoint/playback MoE load;
- no live MuJoCo rollout;
- no GPL implementation copy.

Files:

- Created: `src/unilab/algos/torch/distill/moe_student.py`
  - `MoEStudentPolicy`: router MLP, expert `MLPStudentPolicy` list, soft/hard routing, and diagnostics.
  - `MoEStudentOutput`: action, router logits/probs, expert actions, usage, and selected expert.
- Modified: `src/unilab/algos/torch/distill/__init__.py`
  - Exports `MoEStudentPolicy` and `MoEStudentOutput`.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds semantic toy tests for soft route mixing, hard route selection, router/expert gradients, and bad-contract rejection.

Owner module: `src/unilab/algos/torch/distill/moe_student.py`.

Core parameter path:

```text
student_obs
 -> router(obs) logits
 -> route_probs or hard selected_expert
 -> per-expert MLPStudentPolicy(obs) actions
 -> mixed action
 -> expert_usage diagnostics
 -> router/expert gradients under a toy loss
```

Feature flag contract:

```text
flag name: explicit `MoEStudentPolicy(...)` construction only.
OFF behavior: existing `MLPStudentPolicy`, trainer, checkpoint, and playback paths are unchanged.
ON behavior: caller receives MoE action and diagnostics from the local module.
generated/derived overrides: none yet; config wiring is deferred.
forbidden mixed states: `num_experts < 2`, invalid routing mode, non-positive temperature, and wrong obs dim raise `ValueError`.
tests proving OFF: full generic distillation contract still passes.
tests proving ON: `pytest -k moe` covers soft/hard routing and gradients.
```

Test class: core param path with a tiny semantic fixture.

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k moe
uv run pytest tests/algos/test_g1_distillation_contract.py -q
```

Expected result:

- Soft route with uniform logits mixes three hand-set expert actions into the hand-computable mean action.
- Hard route chooses the highest-bias expert and reports exact usage counts.
- A toy loss backpropagates into both router and expert parameters.
- Existing non-MoE distillation tests keep passing.

Evidence (2026-07-10):

- Before implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q -k moe`: FAIL with missing `MoEStudentPolicy` export.
- After implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q -k moe`: PASS (`3 passed, 10 deselected`).

Status: COMPLETE for local MoE student routing/action/gradient contract only.
Trainer aux loss was still unconfirmed at this step. Config selection,
checkpoint/playback support, live rollout, formal training, and expert role
semantics remain unconfirmed.

### Step 3.2: MoE Trainer Aux-Loss And Usage Diagnostics

Scope: connect `MoEStudentPolicy` diagnostics into the generic behavior
distillation trainer so one offline update can report behavior loss, auxiliary
balance loss, route entropy, expert usage, and router/expert gradients.

Non-scope:

- no Hydra `student.model_type=moe` config selection;
- no checkpoint schema change for MoE architecture;
- no student-only playback loader for MoE checkpoints;
- no live MuJoCo rollout;
- no formal long-running training;
- no expert-role curriculum or standing/walking/recovery labels.

Files:

- Modified: `src/unilab/algos/torch/distill/trainer.py`
  - Adds `aux_loss_coef` to `BehaviorDistillationTrainer`.
  - Keeps default `aux_loss_coef=0.0`, so existing MLP student behavior remains OFF-compatible.
  - Requests `return_diagnostics=True` only when the student accepts it.
  - Computes uniform-load auxiliary loss from `route_probs.mean(dim=0)`.
  - Records `behavior_loss`, `aux_loss`, `expert_usage`, and `route_entropy` in stats.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds a MoE trainer update fixture that checks `loss = behavior_loss + coef * aux_loss`.
  - Checks router and expert gradients are nonzero after the update.

Owner module: `src/unilab/algos/torch/distill/trainer.py`.

Core parameter path:

```text
DistillationBatch.student_obs
 -> student(student_obs, return_diagnostics=True)
 -> MoEStudentOutput.action / route_probs / expert_usage
 -> behavior_loss(student_action, detached_teacher_action)
 -> aux_loss(route_probs.mean(dim=0), uniform_expert_target)
 -> total_loss = behavior_loss + aux_loss_coef * aux_loss
 -> backward()
 -> student_grad_norm, router gradients, expert gradients, stats diagnostics
```

Feature flag contract:

```text
flag name: `aux_loss_coef` on `BehaviorDistillationTrainer`.
OFF behavior: default `aux_loss_coef=0.0`; MLP student path returns the same action/loss contract and no expert diagnostics.
ON behavior: MoE student diagnostics contribute aux loss and stats when `aux_loss_coef > 0`.
generated/derived overrides: none yet; config wiring is deferred.
forbidden mixed states: negative `aux_loss_coef` raises `ValueError`; malformed diagnostic outputs raise explicit errors.
tests proving OFF: full generic distillation contract and script fake-batch tests still pass.
tests proving ON: MoE trainer fixture checks aux loss, usage, entropy, and gradients.
```

Test class: core param path with semantic toy trainer update.

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "moe and trainer"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"
```

Expected result:

- MoE trainer update reports `behavior_loss > 0`, `aux_loss >= 0`, finite route entropy, and two expert usage values.
- `stats.loss == stats.behavior_loss + aux_loss_coef * stats.aux_loss`.
- Router and expert parameter gradients are nonzero.
- Existing MLP-based distillation tests remain passing.

Evidence (2026-07-10):

- Before implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "moe and trainer"` failed with `BehaviorDistillationTrainer.__init__() got an unexpected keyword argument 'aux_loss_coef'`.
- After implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "moe and trainer"`: PASS (`1 passed, 13 deselected`).
- `uv run pytest tests/algos/test_g1_distillation_contract.py -q`: PASS (`14 passed`).
- `uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"`: PASS (`21 passed, 133 deselected`).

Status: COMPLETE for offline MoE trainer aux-loss and diagnostics only.
At this step, config selection, MoE checkpoint/playback support, live rollout,
formal training, and expert role semantics remained unconfirmed; config
selection is addressed later in Step 3.3.

### Step 3.3: MoE Config Selection And Entrypoint Probe

Scope: wire the existing MoE student and trainer aux diagnostics into the
generic `conf/distill` owner config and `scripts/train_distill.py` fake-batch
entrypoint.

Non-scope:

- no student-only playback loader for MoE checkpoints;
- no live MuJoCo rollout for MoE action mode;
- no real replay/storage data path;
- no formal long-running distillation training;
- no expert-role labels or curriculum semantics.

Files:

- Modified: `conf/distill/config.yaml`
  - Adds default-off `student.model_type: mlp`.
  - Adds MoE candidate parameters: `num_experts`, `expert_hidden_dims`,
    `router_hidden_dims`, `routing_mode`, and `router_temperature`.
  - Adds inert default `algo.aux_loss_coef: 0.0`.
- Modified: `scripts/train_distill.py`
  - Builds `MLPStudentPolicy` for `student.model_type=mlp`.
  - Builds `MoEStudentPolicy` for `student.model_type=moe`.
  - Passes `algo.aux_loss_coef` to `BehaviorDistillationTrainer`.
  - Saves student runtime metadata for MLP or MoE fake-batch checkpoints.
  - Returns `behavior_loss`, `aux_loss`, `expert_usage`, and `route_entropy`
    from fake-batch probes.
- Modified: `src/unilab/algos/torch/distill/offline.py`
  - Propagates the latest trainer behavior/aux/route diagnostics into
    `OfflineDistillationRunResult`.
- Modified: tests and notes listed in this step.

Owner flag:

```text
flag name: `student.model_type`
OFF behavior: default `mlp`; old MLP fake-batch checkpoint path still records
  `student_hidden_dims` and remains loadable by student-only playback.
ON behavior: explicit `student.model_type=moe` builds MoE student, activates
  MoE architecture metadata, and reports aux/route diagnostics through the
  fake-batch entrypoint.
generated/derived overrides: `algo.aux_loss_coef` is inert at `0.0`; nonzero
  values affect total loss only after MoE diagnostics expose `route_probs`.
forbidden mixed states: unsupported `student.model_type` raises `ValueError`;
  malformed MoE parameters are rejected by `MoEStudentPolicy`.
tests proving OFF: distill config compose, MLP build, MLP fake-batch checkpoint,
  student-only playback load.
tests proving ON: MoE config compose, MoE build, MoE fake-batch aux diagnostics,
  MoE checkpoint state roundtrip with a matching MoE module.
```

Parameter inventory:

| param | owner | old default | new default | OFF value | ON value | consumers | persistence/playback risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `student.model_type` | `conf/distill/config.yaml` | absent | `mlp` | `mlp` | `moe` | `build_student_policy` | high: playback loader is MLP-only in this step |
| `algo.aux_loss_coef` | `conf/distill/config.yaml` | absent | `0.0` | `0.0` | nonzero override | `BehaviorDistillationTrainer` | low for MLP; MoE training semantics only |
| `student.num_experts` | `conf/distill/config.yaml` | absent | `3` | inert | MoE expert count | `MoEStudentPolicy` | high if checkpoint is loaded into wrong architecture |
| `student.expert_hidden_dims` | `conf/distill/config.yaml` | absent | `[256,256,256]` | inert | expert MLP dims | `MoEStudentPolicy` | high if missing from runtime metadata |
| `student.router_hidden_dims` | `conf/distill/config.yaml` | absent | `[]` | inert | router MLP dims | `MoEStudentPolicy` | high if missing from runtime metadata |
| `student.routing_mode` | `conf/distill/config.yaml` | absent | `soft` | inert | `soft` or `hard` | `MoEStudentPolicy` | medium: affects diagnostics and gradients |
| `student.router_temperature` | `conf/distill/config.yaml` | absent | `1.0` | inert | positive float | `MoEStudentPolicy` | medium: affects route distribution |

Core parameter path:

```text
conf/distill student.model_type
 -> build_student_policy(cfg)
 -> MLPStudentPolicy or MoEStudentPolicy
 -> BehaviorDistillationTrainer(aux_loss_coef)
 -> run_fake_batch_update()
 -> OfflineDistillationRunResult behavior/aux/route diagnostics
 -> distill_runtime_cfg checkpoint metadata
```

Test class: secondary config contract plus core trainer/probe diagnostics path.

Commands:

```bash
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_visualization_entrypoints.py tests/config/test_config_system.py tests/scripts/test_train_scripts.py tests/visualization/test_interactive_playback.py tests/algos/test_g1_distillation_contract.py -q
```

Expected result:

- Default distill config remains `student.model_type=mlp` and `algo.aux_loss_coef=0.0`.
- Explicit MoE overrides compose and build `MoEStudentPolicy`.
- MoE fake-batch probe reports behavior loss, auxiliary loss, expert usage, and route entropy.
- Existing MLP fake-batch checkpoint stays loadable by `load_distillation_student_policy`.

Evidence (2026-07-10):

- Before implementation, distill config tests failed with missing `algo.aux_loss_coef`
  and missing `student.model_type`.
- Before implementation, MoE entrypoint tests failed at Hydra compose because
  `student.model_type` was absent.
- Before implementation, offline result test failed because
  `OfflineDistillationRunResult` had no `last_behavior_loss`.
- After implementation, `uv run pytest tests/config/test_config_system.py -q -k distill`: PASS (`4 passed, 117 deselected`).
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"`: PASS (`23 passed, 133 deselected`).
- After implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q`: PASS (`14 passed`).
- After implementation, touched-suite command above: PASS (`325 passed`).

Status: COMPLETE for config-selectable MoE fake-batch distillation probe.
At this step, MoE student-only playback loader, live MuJoCo MoE policy rollout,
real replay storage, formal training, and expert role semantics remained
unconfirmed; MoE student-only playback loader is addressed later in Step 3.4.

### Step 3.4: MoE Student-Only Playback Loader

Scope: extend generic distillation student-only checkpoint loading so MoE
student checkpoints saved with `distill_runtime_cfg.student_model_type=moe` can
be reconstructed for playback.

Non-scope:

- no trained MoE policy quality claim;
- no viewer-window validation;
- no formal long-running distillation training;
- no real replay/storage dataset integration;
- no expert role labels or curriculum semantics.

Files:

- Modified: `src/unilab/algos/torch/distill/playback.py`
  - Defaults missing `student_model_type` to `mlp` for old MLP checkpoints.
  - Rebuilds `MLPStudentPolicy` from legacy MLP metadata.
  - Rebuilds `MoEStudentPolicy` from MoE runtime metadata.
  - Raises `ValueError` for unsupported `student_model_type`.
- Modified: `scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py`
  - Adds `--temp-student-model-type {mlp,moe}` for temp checkpoint construction.
  - Keeps default temp checkpoint type as MLP.
  - Saves MoE runtime metadata when building a temp MoE checkpoint.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds MoE student-only playback load contract.
  - Adds unknown model-type rejection.
  - Keeps MLP legacy load contract.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds fake-session sentinel contract for temp MoE checkpoint creation and load.

Owner flag:

```text
flag name: checkpoint metadata `distill_runtime_cfg.student_model_type`
OFF behavior: missing or `mlp` reconstructs `MLPStudentPolicy` exactly as before.
ON behavior: `moe` reconstructs `MoEStudentPolicy` from saved architecture metadata.
generated/derived overrides: sentinel-only `--temp-student-model-type` controls temp checkpoint generation, default `mlp`.
forbidden mixed states: unsupported model type raises `ValueError`; MoE checkpoints missing required architecture metadata fail before state-dict load.
tests proving OFF: MLP student-only playback load contract and existing distill script tests.
tests proving ON: MoE student-only playback load contract, sentinel temp MoE checkpoint fake-session contract, and 1-step temp MoE live sentinel.
```

Parameter inventory:

| param | owner | old default | new default | OFF value | ON value | consumers | persistence/playback risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `distill_runtime_cfg.student_model_type` | checkpoint metadata | absent | absent in old runs | missing or `mlp` | `moe` | `load_distillation_student_policy` | high: decides policy class before state-dict load |
| `student_num_experts` | checkpoint metadata | absent | required for MoE | inert | int >= 2 | `MoEStudentPolicy` playback reconstruction | high if missing or mismatched |
| `student_expert_hidden_dims` | checkpoint metadata | absent | required for MoE | inert | list[int] | `MoEStudentPolicy` expert MLPs | high if mismatched |
| `student_router_hidden_dims` | checkpoint metadata | absent | required for MoE | inert | list[int] | `MoEStudentPolicy` router MLP | high if mismatched |
| `student_routing_mode` | checkpoint metadata | absent | `soft` fallback | inert | `soft` or `hard` | `MoEStudentPolicy` action route | medium |
| `student_router_temperature` | checkpoint metadata | absent | `1.0` fallback | inert | positive float | `MoEStudentPolicy` action route | medium |
| `--temp-student-model-type` | sentinel CLI | absent | `mlp` | `mlp` | `moe` | live sentinel temp checkpoint factory | low, sentinel-only |

Core parameter path:

```text
checkpoint["distill_runtime_cfg"].student_model_type
 -> load_distillation_student_policy()
 -> MLPStudentPolicy or MoEStudentPolicy reconstruction
 -> load_distillation_checkpoint(policy, strict=True)
 -> LoadedDistillationStudentPolicy.policy(obs)
 -> create_distill_playback_session policy callable
```

Test class: checkpoint/playback persistence contract.

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "moe_student_checkpoint_loads_for_student_only_playback or unknown_model_type or student_checkpoint_loads_for_student_only_playback"
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill_playback_live_sentinel_policy_checkpoint_contract or moe_policy_checkpoint_contract"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"
uv run pytest tests/visualization/test_interactive_playback.py -q -k distill
uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --steps 1 --action-mode policy --make-temp-policy-checkpoint --temp-student-model-type moe --device cpu
```

Expected result:

- Legacy MLP student checkpoints still load without `student_model_type`.
- MoE checkpoints reconstruct `MoEStudentPolicy`, freeze gradients, and emit finite 29-D deployable actions for 99-D actor observations.
- Unsupported `student_model_type` fails with explicit `ValueError`.
- Sentinel temp MoE checkpoint can be generated and loaded through the same student-only loader in a fake-session contract.
- The real `G1WalkHeight` MuJoCo playback path steps once with a temp MoE checkpoint and emits finite nonzero 29-D actions from 99-D actor observations.

Evidence (2026-07-10):

- Before implementation, MoE playback load failed by trying to load MoE state dict into `MLPStudentPolicy`.
- Before implementation, unknown model type fell through to MLP and failed with state-dict mismatch instead of explicit model-type error.
- Before implementation, sentinel test failed with `run_check() got an unexpected keyword argument 'temp_student_model_type'`.
- After implementation, targeted algos playback tests: PASS (`3 passed, 13 deselected`).
- After implementation, targeted sentinel tests: PASS (`2 passed, 155 deselected`).
- After implementation, `uv run pytest tests/algos/test_g1_distillation_contract.py -q`: PASS (`16 passed`).
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"`: PASS (`24 passed, 133 deselected`).
- After implementation, `uv run pytest tests/visualization/test_interactive_playback.py -q -k distill`: PASS (`5 passed, 12 deselected`).
- After implementation, touched-suite command: PASS (`328 passed`).
- After implementation, MoE temp-checkpoint live sentinel: PASS, with `policy_obs_mode=actor`, `action_dim=29`, `physics_shape=[1,72]`, `actions_shape=[1,29]`, and `actions_abs_max=0.049958`.

Status: COMPLETE for MoE student-only checkpoint playback loader and sentinel
temp-checkpoint factory, including a 1-step real MuJoCo playback sentinel for a
temporary MoE checkpoint. Viewer-window validation, trained MoE policy quality,
real replay/storage integration, formal training, and expert role semantics
remain unconfirmed.

### Step 3.5: Saved Dataset Offline Update Entrypoint

Scope: connect saved `DistillationTensorDataset` files into the generic
`scripts/train_distill.py` entrypoint so a bounded offline update can run from
persisted student/teacher observation tensors.

Non-scope:

- no replay collector or live env sampler;
- no formal long-running distillation training;
- no trained policy quality claim;
- no viewer-window validation;
- no expert role labels or curriculum semantics.

Files:

- Modified: `conf/distill/config.yaml`
  - Adds inert defaults: `training.offline_dataset_path: null`,
    `offline_batch_size: 256`, `offline_max_updates: 1`, and
    `offline_checkpoint: null`.
- Modified: `scripts/train_distill.py`
  - Adds `run_offline_dataset_update()`.
  - Loads saved datasets through `load_distillation_dataset()` with configured
    teacher/student obs-dim guards.
  - Saves runtime metadata with `distill_source=offline_dataset` and
    `dataset_path`.
  - Routes Hydra main to saved-dataset offline update when
    `training.offline_dataset_path` is set.
- Modified: `tests/config/test_config_system.py`
  - Checks offline dataset defaults and explicit overrides.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds a saved-dataset MoE offline update fixture that saves a student
    checkpoint and reloads it through student-only playback.

Owner flag:

```text
flag name: `training.offline_dataset_path`
OFF behavior: `null`; existing fake-batch dry-run path and NotImplemented live-training guard remain unchanged.
ON behavior: load the saved dataset, run bounded offline updates, and optionally save a student checkpoint.
generated/derived overrides: `offline_batch_size`, `offline_max_updates`, and `offline_checkpoint` are consumed only by this path.
forbidden mixed states: dataset obs dims must match configured `student.obs_dim` and `teacher.obs_dim`; mismatches fail in `load_distillation_dataset`.
tests proving OFF: distill config compose and existing fake-batch/script tests.
tests proving ON: saved-dataset MoE update fixture and real Hydra CLI probe.
```

Parameter inventory:

| param | owner | old default | new default | OFF value | ON value | consumers | persistence/playback risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `training.offline_dataset_path` | `conf/distill/config.yaml` | absent | `null` | `null` | dataset file path | `run_offline_dataset_update`, Hydra main | high: opens saved dataset path |
| `training.offline_batch_size` | `conf/distill/config.yaml` | absent | `256` | inert | positive int | `run_offline_distillation_updates` | medium |
| `training.offline_max_updates` | `conf/distill/config.yaml` | absent | `1` | inert | positive int | `run_offline_distillation_updates` | medium |
| `training.offline_checkpoint` | `conf/distill/config.yaml` | absent | `null` | no checkpoint | checkpoint path | `save_distillation_checkpoint` | high for playback |
| `distill_runtime_cfg.distill_source` | checkpoint metadata | absent | `offline_dataset` on this path | absent/fake | `offline_dataset` | playback/audit metadata | low |
| `distill_runtime_cfg.dataset_path` | checkpoint metadata | absent | source path on this path | absent | dataset path | audit/debug metadata | medium |

Core parameter path:

```text
training.offline_dataset_path
 -> load_distillation_dataset(expected student=99, teacher=100)
 -> DistillationTensorDataset.as_batch()
 -> BehaviorDistillationTrainer.update()
 -> run_offline_distillation_updates()
 -> save_distillation_checkpoint()
 -> load_distillation_student_policy()
```

Test class: checkpoint/storage persistence contract with a saved tensor dataset.

Commands:

```bash
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/scripts/test_train_scripts.py -q -k dataset_update_loads_saved_dataset
uv run scripts/train_distill.py teacher.load_run=/private/tmp/unilab-step15/teacher_run teacher.actor_hidden_dim=16 teacher.use_layer_norm=false teacher.obs_normalization=false student.model_type=moe student.num_experts=3 'student.expert_hidden_dims=[32]' 'student.router_hidden_dims=[16]' algo.learning_rate=0.01 algo.max_grad_norm=10.0 algo.aux_loss_coef=0.25 training.offline_dataset_path=/private/tmp/unilab-step15/dataset.pt training.offline_batch_size=2 training.offline_max_updates=2 training.offline_checkpoint=/private/tmp/unilab-step15/offline_moe_student.pt
uv run python -c "import torch; from unilab.algos.torch.distill import load_distillation_student_policy; loaded=load_distillation_student_policy('/private/tmp/unilab-step15/offline_moe_student.pt', device='cpu'); action=loaded.policy(torch.randn(1,99)); print(type(loaded.policy).__name__, loaded.obs_dim, loaded.action_dim, tuple(action.shape), bool(torch.isfinite(action).all()))"
```

Expected result:

- OFF defaults compose with `offline_dataset_path=None`.
- ON overrides compose and activate all saved-dataset parameters together.
- Saved dataset path loads with the configured student/teacher obs dims. At
  the time of Step 3.5 this was 99-D student and 100-D teacher; Step 3.8
  changes the current default to 99-D teacher while preserving 100-D as an
  explicit legacy override.
- MoE offline update reports finite behavior loss, aux loss, expert usage, and route entropy.
- Saved MoE student checkpoint reloads through student-only playback and emits finite `(1,29)` actions.

Evidence (2026-07-10):

- Before implementation, config tests failed with missing `training.offline_dataset_path`.
- Before implementation, script test failed with missing `run_offline_dataset_update`.
- After implementation, `uv run pytest tests/config/test_config_system.py -q -k distill`: PASS (`4 passed, 117 deselected`).
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k dataset_update_loads_saved_dataset`: PASS (`1 passed, 157 deselected`).
- After implementation, direct `scripts/train_distill.py` Hydra CLI probe: PASS, with `distill_source=offline_dataset`, `student_model_type=moe`, `dataset_num_samples=4`, `update_count=2`, `samples_seen=4`, finite `loss=0.028123`, finite `aux_loss=0.017329`, and three expert usage values.
- After implementation, student-only reload probe: PASS (`MoEStudentPolicy 99 29 (1, 29) True`).
- After implementation, `uv run pytest tests/scripts/test_train_scripts.py -q -k "distill or distill_playback_live_sentinel"`: PASS (`25 passed, 133 deselected`).
- After implementation, touched-suite command: PASS (`329 passed`).

Status: COMPLETE for saved-dataset offline update wiring and MoE checkpoint
roundtrip. Replay collector/live dataset generation, formal long-running
training, trained MoE policy quality, viewer-window validation, and expert role
semantics remain unconfirmed.

### Step 3.6: Live Env Dataset Collection Entrypoint

Scope: add the smallest UniLab-owned live/env observation collector that can
save a `DistillationTensorDataset` from `G1WalkHeight` without starting formal
distillation training.

Non-scope:

- no replay buffer integration;
- no teacher action quality claim;
- no formal long-running training;
- no viewer-window validation;
- no expert role labels or curriculum semantics.

Files:

- Added: `src/unilab/algos/torch/distill/collector.py`
  - Owns env obs extraction, teacher projection, student projection, zero-action
    rollout sampling, finite/shape guards, and dataset metadata.
- Modified: `src/unilab/algos/torch/distill/__init__.py`
  - Exports collector helpers.
- Modified: `conf/distill/config.yaml`
  - Adds inert defaults for `training.collect_dataset_path`,
    `collect_num_samples`, `collect_num_envs`, `collect_action_mode`,
    `collect_teacher_obs_key`, `collect_teacher_projection`,
    `collect_student_projection`, and `collect_student_drop_index`.
- Modified: `scripts/train_distill.py`
  - Adds `run_collect_dataset()` as a thin assembly route.
  - Routes Hydra main to collection only when `training.collect_dataset_path` is
    non-null.
- Modified: tests and notes.

Owner flag:

```text
flag name: `training.collect_dataset_path`
OFF behavior: `null`; no live env collection, saved-dataset update and dry-run behavior remain unchanged.
ON behavior: create the configured env, collect bounded observation samples, save a `DistillationTensorDataset`, and print shape/projection facts.
generated/derived overrides: `collect_num_samples`, `collect_num_envs`, `collect_action_mode`, `collect_teacher_obs_key`, `collect_teacher_projection`, `collect_student_projection`, `collect_student_drop_index`.
forbidden mixed states: unsupported action modes fail; missing drop index fails when student projection is `drop_index`; projection dims must match configured student/teacher dims.
tests proving OFF: distill config compose with `collect_dataset_path=None`.
tests proving ON: fake-env script contract and 1-sample live MuJoCo CLI sentinel.
```

Parameter inventory:

| param | owner | old default | new default | OFF value | ON value | consumers | persistence/playback risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `training.collect_dataset_path` | `conf/distill/config.yaml` | absent | `null` | no collection | output dataset path | `run_collect_dataset`, Hydra main | high: writes dataset file |
| `training.collect_num_samples` | `conf/distill/config.yaml` | absent | `1024` | inert | positive int | collector loop | medium |
| `training.collect_num_envs` | `conf/distill/config.yaml` | absent | `1` | inert | positive int | env factory | medium |
| `training.collect_action_mode` | `conf/distill/config.yaml` | absent | `zero` | inert | `zero` | collector action source | medium |
| `training.collect_teacher_obs_key` | `conf/distill/config.yaml` | absent | `obs` | inert | env obs key | collector obs extraction | high |
| `training.collect_teacher_projection` | `conf/distill/config.yaml` | absent | `identity` | inert | `identity` by default, `pad_zeros` only with explicit legacy teacher dim override | collector teacher projection | high: `pad_zeros` is synthetic tail |
| `training.collect_student_projection` | `conf/distill/config.yaml` | absent | `identity` | inert | `identity` or `drop_index` | collector student projection | high |
| `training.collect_student_drop_index` | `conf/distill/config.yaml` | absent | `null` | inert | int only for `drop_index` | collector student projection | high |

Core parameter path:

```text
training.collect_dataset_path
 -> create_env(task_name=G1WalkHeight, sim_backend=mujoco)
 -> env.reset()
 -> obs["obs"] source tensor
 -> teacher_projection=identity -> teacher_obs 99-D
 -> student_projection=identity -> student_obs 99-D
 -> build_distillation_dataset()
 -> save_distillation_dataset()
 -> load_distillation_dataset(expected student=99, teacher=100)
```

Runtime fact discovered:

- Historical Step 16 fact: live `G1WalkHeight` created from `conf/distill` reported
  `obs_groups_spec={'obs': 99, 'critic': 102}` and `obs['obs'].shape=(1,99)`,
  while the frozen SAC teacher checkpoint contract remains `teacher.obs_dim=100`.
- Therefore Step 16 uses `teacher_projection=pad_zeros` and records
  `synthetic_teacher_tail=True`. This proves storage/connectivity only; it does
  not prove true teacher privileged semantics.
  Step 3.8 supersedes the default with 99-D identity and keeps this bridge only
  as a legacy override.

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k collect_distillation_dataset_from_env
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/scripts/test_train_scripts.py -q -k collects_live_env_dataset
uv run scripts/train_distill.py training.collect_dataset_path=/private/tmp/unilab-step16-live-dataset.pt training.collect_num_samples=1 training.collect_num_envs=1
uv run python -c "from unilab.algos.torch.distill import load_distillation_dataset; d=load_distillation_dataset('/private/tmp/unilab-step16-live-dataset.pt', expected_student_obs_dim=99, expected_teacher_obs_dim=100); print(d.num_samples, d.student_obs_dim, d.teacher_obs_dim, d.metadata['source'], d.metadata['synthetic_teacher_tail'])"
```

Evidence (2026-07-10):

- Before implementation, collector tests failed with missing
  `collect_distillation_dataset_from_env`.
- Before implementation, config/script tests failed with missing
  `training.collect_*` fields.
- First live attempt with `student_projection=drop_index` failed:
  `student projection dim mismatch: expected 99, got 98`.
- Live probe showed current env fact:
  `spec {'obs': 99, 'critic': 102}`, `student_dim 99`, `teacher_dim 100`.
- After implementation, collector contract tests: PASS (`3 passed, 16 deselected`).
- After implementation, distill config tests: PASS (`4 passed, 117 deselected`).
- After implementation, script fake-env collect test: PASS (`1 passed, 158 deselected`).
- After implementation, live MuJoCo collection CLI: PASS with
  `dataset_num_samples=1`, `dataset_student_obs_dim=99`,
  `dataset_teacher_obs_dim=100`, `teacher_projection=pad_zeros`,
  `student_projection=identity`, and `synthetic_teacher_tail=True`.
- Saved live dataset reload: PASS (`1 99 100 live_env_rollout True`).

Status: COMPLETE for live-env observation dataset collection and persistence
only. Superseded by Step 3.8 for the current default teacher obs dimension:
the default collector now saves 99-D teacher obs with identity projection, and
the 100-D pad bridge is legacy override only. Replay storage, nonzero/policy
action sampling, real teacher privileged semantics, formal training, trained
policy quality, viewer-window validation, and expert role semantics remain
unconfirmed.

### Step 3.7: Teacher Observation Contract Audit

Scope: audit whether the generic G1 distillation teacher observation dimension
is a true live-env semantic contract or only a checkpoint/storage compatibility
bridge.

Non-scope:

- no formal training;
- no teacher action quality claim;
- no replay collector;
- no checkpoint migration;
- no formal training-loop migration;
- no forced selection of a 100-D external checkpoint.

Files:

- Added: `scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py`
  - Reports config teacher/student dims, live env obs spec, reset obs shapes,
    projection bridge status, and optional checkpoint actor input dim.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds fake-env contract coverage for the 99-D default, the explicit
    99-D live obs -> 100-D legacy `pad_zeros` bridge, and checkpoint
    first-layer input inspection.
- Modified: testing notes.

Core parameter path:

```text
conf/distill teacher.obs_dim
 -> G1WalkHeight env.obs_groups_spec
 -> live obs["obs"] shape
 -> collect_teacher_projection
 -> optional checkpoint actor first weight input dim
```

Runtime facts discovered:

- Step 17 fact: `conf/distill` composed `teacher.obs_dim=100`,
  `student.obs_dim=99`, `collect_teacher_projection=pad_zeros`, and
  `env.commands.observe_height_command=true` with no `env.mode_observation`.
- Current live `G1WalkHeight` MuJoCo env reports
  `obs_groups_spec={'obs': 99, 'critic': 102}` and reset produces
  `obs['obs'].shape=(1,99)`.
- Local `logs/fast_sac/G1WalkFlat/2026-07-08_*` checkpoints are mixed:
  some actor first layers are 100-D, later runs are 99-D or 98-D, and these
  directories do not carry `run_config.json`; no local
  `logs/fast_sac/G1WalkHeight` teacher checkpoint was found.
- Therefore the current 100-D teacher route is checkpoint-compatible only when
  the selected teacher checkpoint is actually 100-D. The live collector's
  `pad_zeros` bridge proves storage/connectivity, not real teacher privileged
  semantics.

Commands:

```bash
uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k teacher_obs_contract
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py --checkpoint-path logs/fast_sac/G1WalkFlat/2026-07-08_15-42-50_mujoco/model_5000.pt
```

Evidence (2026-07-10):

- `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q`:
  PASS (`14 passed`).
- Live env probe: PASS with `obs_groups_spec={'obs': 99, 'critic': 102}`,
  `obs_shape=(1,99)`, and `critic_shape=(1,102)`.
- Live collection probe: PASS with `dataset_student_obs_dim=99`,
  `dataset_teacher_obs_dim=100`, and `synthetic_teacher_tail=True`.
- `uv run pytest tests/scripts/test_train_scripts.py -q -k teacher_obs_contract`:
  PASS (`2 passed, 159 deselected`).
- `uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py`:
  PASS with no FAIL checks, `student_live_dim=99`, reset shapes
  `obs=(1,99)` and `critic=(1,102)`, plus WARN checks for
  `projection_bridge` and uninspected checkpoint input dim.
- `uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py --checkpoint-path logs/fast_sac/G1WalkFlat/2026-07-08_15-42-50_mujoco/model_5000.pt`:
  PASS with `checkpoint_first_weight=net.0.weight`,
  `checkpoint_input_dim=100`, and the same WARN `projection_bridge`.
- Checkpoint input-dim probe found a 100-D local WalkFlat checkpoint, but no
  local WalkHeight run metadata proving that this is the intended height
  teacher.

Status: COMPLETE for teacher obs contract audit and diagnostic sentinel. Step
3.8 implements the semantic fix by migrating the default distill teacher
contract to the current 99-D `G1WalkHeight` live env layout and switching
collection to identity; 100-D remains explicit legacy override only.

### Step 3.8: Default Teacher Obs Contract Alignment

Scope: make the generic G1 distillation default teacher observation contract
match the current live `G1WalkHeight` actor observation layout.

Non-scope:

- no formal distillation training loop;
- no replay collector integration;
- no trained policy quality claim;
- no removal of 100-D checkpoint compatibility when explicitly requested.

Files:

- Modified: `conf/distill/config.yaml`
  - Changes default `teacher.obs_dim` from 100 to 99.
  - Changes default `training.collect_teacher_projection` from `pad_zeros` to
    `identity`.
- Modified: `conf/distill/task/g1_walk_height/mujoco.yaml`
  - Changes task-owner teacher obs dim to 99 so the task group no longer
    reopens the old 100-D default.
- Modified: `tests/config/test_config_system.py`
  - Guards default 99/99 teacher/student dims and explicit 100-D legacy
    projection override.
- Modified: `tests/scripts/test_train_scripts.py`
  - Updates fake-batch/offline-dataset fixtures to 99-D teacher by default.
  - Keeps a legacy bridge test for `teacher.obs_dim=100` plus
    `training.collect_teacher_projection=pad_zeros`.
- Modified: testing notes.

Owner flag / config group:

```text
flag name: `conf/distill` + optional legacy override
OFF behavior: non-distill offpolicy/playback paths do not read this config.
ON default behavior: `teacher.obs_dim=99`, `student.obs_dim=99`,
  `collect_teacher_projection=identity`, matching live `G1WalkHeight` actor obs.
legacy ON override: `teacher.obs_dim=100` plus
  `training.collect_teacher_projection=pad_zeros` keeps old checkpoint bridge
  explicit and auditable.
forbidden mixed states: `teacher.obs_dim=100` with `identity` projection fails
  on live 99-D obs; mismatched checkpoint first layer fails through dim guard.
tests proving OFF: offpolicy/height obs tests still pass outside distill.
tests proving ON: config, script, collector, and live collection probes below.
```

Parameter inventory:

| param | owner | old default | new default | OFF value | ON value | consumers | persistence/playback risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `teacher.obs_dim` | `conf/distill/*` | 100 | 99 | absent outside distill | 99 default, 100 explicit legacy | teacher loader, fake dataset, saved dataset loader | high, checkpoint input dim |
| `training.collect_teacher_projection` | `conf/distill/config.yaml` | `pad_zeros` | `identity` | inert | `identity` default, `pad_zeros` explicit legacy | live collector | high, synthetic tail |
| `training.collect_dataset_path` | `conf/distill/config.yaml` | `null` | `null` | no collection | output dataset path | `run_collect_dataset` | high, writes dataset |

Core parameter path:

```text
conf/distill task owner
 -> teacher.obs_dim=99
 -> run_collect_dataset()
 -> collect_distillation_dataset_from_env()
 -> project_teacher_obs(identity)
 -> DistillationTensorDataset teacher_obs_dim=99
 -> load_distillation_dataset(expected_teacher_obs_dim=99)
```

Evidence (2026-07-10):

- First targeted config/script test run failed because
  `conf/distill/task/g1_walk_height/mujoco.yaml` still overrode
  `teacher.obs_dim=100`; this identified the true owner-layer stale default.
- After updating the task owner, `uv run pytest tests/config/test_config_system.py -q -k distill`:
  PASS (`4 passed, 117 deselected`).
- `uv run pytest tests/scripts/test_train_scripts.py -q -k "distill and not hora"`:
  PASS (`13 passed, 149 deselected`).
- Default live collection:
  `uv run scripts/train_distill.py training.collect_dataset_path=/private/tmp/unilab-step18-default-live-dataset.pt training.collect_num_samples=1 training.collect_num_envs=1`
  PASS with `dataset_student_obs_dim=99`, `dataset_teacher_obs_dim=99`,
  `teacher_projection=identity`, and `synthetic_teacher_tail=False`.
- Explicit legacy collection:
  `uv run scripts/train_distill.py teacher.obs_dim=100 training.collect_teacher_projection=pad_zeros training.collect_dataset_path=/private/tmp/unilab-step18-legacy100-live-dataset.pt training.collect_num_samples=1 training.collect_num_envs=1`
  PASS with `dataset_student_obs_dim=99`, `dataset_teacher_obs_dim=100`,
  `teacher_projection=pad_zeros`, and `synthetic_teacher_tail=True`.
- Default teacher audit sentinel:
  `uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py`
  PASS with `teacher_obs_dim=99`, `student_obs_dim=99`,
  `teacher_projection=identity`, live `obs_groups_spec={'obs': 99,
  'critic': 102}`, reset `obs=(1,99)`, reset `critic=(1,102)`, and
  `teacher_live_dim=99`; checkpoint input dim remains WARN until a real
  teacher checkpoint is supplied.

Status: COMPLETE for default teacher obs contract alignment. Real trained
teacher quality, replay storage, formal training, and 100-D external checkpoint
semantics remain separate live-only or checkpoint-specific boundaries.

### Step 3.9: Teacher Checkpoint Dim Preflight

Scope: add a distill-owned checkpoint preflight so teacher checkpoint input
dimensions are diagnosed before `load_state_dict` reaches a generic shape
mismatch.

Non-scope:

- no automatic 99-D vs 100-D guessing;
- no checkpoint migration or rewrite;
- no formal training loop;
- no teacher quality claim.

Files:

- Modified: `src/unilab/algos/torch/distill/teacher.py`
  - Adds `DistillationTeacherCheckpointInfo`.
  - Adds `inspect_sac_teacher_checkpoint()`.
  - Adds `validate_sac_teacher_checkpoint_contract()`.
  - Calls the validation before `load_sac_teacher_policy()` constructs and
    loads the actor.
- Modified: `src/unilab/algos/torch/distill/__init__.py`
  - Exports the checkpoint inspect/validate helpers.
- Modified: `scripts/train_distill.py`
  - Calls the validation before building the distillation trainer.
  - Records `checkpoint_actor_input_dim` and `checkpoint_first_weight_key` in
    teacher metadata.
- Modified: `scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py`
  - Reuses the shared checkpoint inspector.
- Modified: tests and notes.

Owner flag / config group:

```text
flag name: `teacher.obs_dim`
OFF behavior: no distill teacher load outside `conf/distill`.
ON default behavior: 99-D teacher checkpoint required.
legacy ON behavior: 100-D teacher checkpoint requires explicit
  `teacher.obs_dim=100` and, for live collection, explicit
  `training.collect_teacher_projection=pad_zeros`.
forbidden mixed states: default 99-D config with a 100-D checkpoint fails
  before trainer construction with a distill-specific diagnostic.
tests proving OFF: non-distill suites remain unaffected by import/export only.
tests proving ON: checkpoint inspector, mismatch rejection, default/legacy
  fake-update probe.
```

Core parameter path:

```text
checkpoint["actor"][first rank-2 weight].shape[1]
 -> DistillationTeacherCheckpointInfo.actor_input_dim
 -> compare with cfg.teacher.obs_dim
 -> fail with override hint or proceed to load_sac_teacher_policy()
 -> teacher_metadata checkpoint_actor_input_dim
```

Evidence (2026-07-10):

- `uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "teacher_checkpoint"`:
  PASS (`3 passed, 17 deselected`).
- `uv run pytest tests/scripts/test_train_scripts.py -q -k "distill_script_fake_batch_probe_loads_teacher or rejects_legacy_teacher_checkpoint_without_override or teacher_obs_contract"`:
  PASS (`5 passed, 158 deselected`).
- Runtime preflight probe with a synthetic 100-D SAC teacher checkpoint:
  default 99-D config rejected with
  `SAC teacher checkpoint obs dim mismatch: checkpoint actor input dim=100
  (net.0.weight), configured teacher.obs_dim=99`, and the error text included
  the explicit `teacher.obs_dim=100` legacy override hint.
- The same 100-D checkpoint with explicit legacy overrides completed one fake
  update: `teacher_obs_shape=(1,100)`, `teacher_action_shape=(1,29)`,
  `update_count=1`, `teacher_action_requires_grad=False`.
- `uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py`:
  PASS for the default 99-D live path, with `teacher_live_dim=99`.
- `uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py --checkpoint-path /private/tmp/unilab-step19/teacher_100d.pt`:
  expected FAIL, reporting `checkpoint_input_dim: expected=99 got=100
  key=net.0.weight`.

Status: COMPLETE for teacher checkpoint input-dim preflight and diagnostics.
Remaining unconfirmed boundaries are real teacher checkpoint quality, formal
training, replay/storage integration, and external 100-D checkpoint semantics.

### Step 3.10: Default 99-D Hydra CLI Dry-Run And Offline Update

Scope: prove the real Hydra `scripts/train_distill.py` entrypoint can use the
current default 99-D teacher contract for both fake-batch dry-run and saved
dataset offline update, including student checkpoint save/load.

Non-scope:

- no replay collector integration;
- no formal long training loop;
- no teacher quality or student policy quality claim;
- no viewer-window validation.

Files:

- No code changes.
- Updated testing and semantic notes with the runtime proof boundary.

Owner module:

```text
scripts/train_distill.py
 -> resolve_teacher_checkpoint()
 -> validate_sac_teacher_checkpoint_contract()
 -> build_distillation_trainer()
 -> run_fake_batch_update() / run_offline_dataset_update()
 -> save_distillation_checkpoint()
 -> load_distillation_student_policy()
```

Core parameter path:

```text
teacher.load_run=/private/tmp/unilab-step20/teacher_99d.pt
 -> checkpoint actor first rank-2 weight input dim=99
 -> cfg.teacher.obs_dim=99
 -> dataset teacher_obs_dim=99
 -> teacher_action_shape=(batch,29)
 -> student checkpoint metadata checkpoint_actor_input_dim=99
 -> deployable student reload obs_dim=99, action_shape=(1,29)
```

Commands:

```bash
uv run scripts/train_distill.py training.dry_run=true teacher.load_run=/private/tmp/unilab-step20/teacher_99d.pt teacher.checkpoint=-1 teacher.actor_hidden_dim=16 teacher.use_layer_norm=false teacher.obs_normalization=false 'student.hidden_dims=[32]' algo.learning_rate=0.01 algo.max_grad_norm=10.0 training.dry_run_batch_size=2 training.dry_run_updates=2 training.dry_run_checkpoint=/private/tmp/unilab-step20/default99_dryrun_student.pt
uv run scripts/train_distill.py teacher.load_run=/private/tmp/unilab-step20/teacher_99d.pt teacher.checkpoint=-1 teacher.actor_hidden_dim=16 teacher.use_layer_norm=false teacher.obs_normalization=false student.model_type=moe student.num_experts=3 'student.expert_hidden_dims=[32]' 'student.router_hidden_dims=[16]' algo.learning_rate=0.01 algo.max_grad_norm=10.0 algo.aux_loss_coef=0.25 training.offline_dataset_path=/private/tmp/unilab-step20/default99_dataset.pt training.offline_batch_size=2 training.offline_max_updates=2 training.offline_checkpoint=/private/tmp/unilab-step20/default99_offline_moe_student.pt
uv run python - <<'PY'
from pathlib import Path
import torch
from unilab.algos.torch.distill.playback import load_distillation_student_policy

for path in [
    Path('/private/tmp/unilab-step20/default99_dryrun_student.pt'),
    Path('/private/tmp/unilab-step20/default99_offline_moe_student.pt'),
]:
    loaded = load_distillation_student_policy(path, device='cpu')
    with torch.no_grad():
        action = loaded.policy(torch.zeros(1, loaded.obs_dim))
    print(path, loaded.obs_dim, tuple(action.shape),
          loaded.teacher_metadata.get('checkpoint_actor_input_dim'))
PY
```

Evidence (2026-07-10):

- Prepared a temporary 99-D SAC teacher checkpoint and a saved 99-D/99-D
  `DistillationTensorDataset` under `/private/tmp/unilab-step20`.
- Default dry-run Hydra CLI: PASS with `student_obs_shape=(2,99)`,
  `teacher_obs_shape=(2,99)`, `dataset_teacher_obs_dim=99`,
  `teacher_action_shape=(2,29)`, `teacher_action_requires_grad=False`,
  `update_count=2`, and checkpoint
  `/private/tmp/unilab-step20/default99_dryrun_student.pt`.
- Saved-dataset offline MoE Hydra CLI: PASS with
  `student_model_type='moe'`, `dataset_teacher_obs_dim=99`,
  finite `loss=0.028084291145205498`, nonzero `aux_loss`,
  populated `expert_usage`, `route_entropy=1.0823466777801514`,
  `update_count=2`, and checkpoint
  `/private/tmp/unilab-step20/default99_offline_moe_student.pt`.
- Student checkpoint reload probe: PASS for both MLP dry-run and MoE offline
  checkpoints with `obs_dim=99`, `action_shape=(1,29)`,
  `teacher_obs_dim=99`, and `checkpoint_actor_input_dim=99`.

Status: COMPLETE for default 99-D Hydra CLI dry-run, saved-dataset offline
update, and student-only checkpoint reload. Remaining unconfirmed boundaries
are replay/storage integration, formal long training, real teacher quality,
trained student quality, live policy rollout from a trained distill checkpoint,
and viewer-window validation.

### Step 3.11: Live Dataset To MoE Student Playback Sentinel

Scope: prove the shortest real-data distillation chain reaches deployable
playback:

```text
real G1WalkHeight MuJoCo reset/step
 -> live 99-D/99-D DistillationTensorDataset
 -> bounded MoE offline update
 -> saved distill student checkpoint
 -> real G1WalkHeight MuJoCo policy playback step
```

Non-scope:

- no replay collector integration;
- no long training loop;
- no teacher or student policy quality claim;
- no viewer-window validation;
- no statement that the random temporary teacher is useful for control.

Files:

- No code changes.
- Updated testing and semantic notes with the runtime proof boundary.

Owner module:

```text
scripts/train_distill.py collection/offline modes
 -> src/unilab/algos/torch/distill dataset/trainer/checkpoint/playback
 -> scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py
```

Core parameter path:

```text
live env obs["obs"] shape=(2,99)
 -> saved dataset student_obs_dim=99, teacher_obs_dim=99
 -> MoE offline update teacher_action_shape=(1,29)
 -> saved checkpoint student_model_type=moe
 -> playback loads /private/tmp/unilab-step21/live_moe_student.pt
 -> policy action shape=(1,29), actions_abs_max=0.2374802529811859
```

Commands:

```bash
uv run scripts/train_distill.py training.collect_dataset_path=/private/tmp/unilab-step21/live_dataset.pt training.collect_num_samples=2 training.collect_num_envs=1
uv run scripts/train_distill.py teacher.load_run=/private/tmp/unilab-step21/teacher_99d.pt teacher.checkpoint=-1 teacher.actor_hidden_dim=16 teacher.use_layer_norm=false teacher.obs_normalization=false student.model_type=moe student.num_experts=3 'student.expert_hidden_dims=[32]' 'student.router_hidden_dims=[16]' algo.learning_rate=0.01 algo.max_grad_norm=10.0 algo.aux_loss_coef=0.25 training.offline_dataset_path=/private/tmp/unilab-step21/live_dataset.pt training.offline_batch_size=1 training.offline_max_updates=2 training.offline_checkpoint=/private/tmp/unilab-step21/live_moe_student.pt
uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --steps 1 --action-mode policy --load-run /private/tmp/unilab-step21/live_moe_student.pt --device cpu
```

Evidence (2026-07-10):

- Temporary 99-D SAC teacher checkpoint prepared at
  `/private/tmp/unilab-step21/teacher_99d.pt`.
- Live collection CLI: PASS with `dataset_num_samples=2`,
  `dataset_student_obs_dim=99`, `dataset_teacher_obs_dim=99`,
  `teacher_projection=identity`, `student_projection=identity`,
  `synthetic_teacher_tail=False`, and `env_steps=1`.
- MoE offline update from the live dataset: PASS with
  `student_model_type='moe'`, `teacher_action_shape=(1,29)`,
  `teacher_action_requires_grad=False`, finite
  `loss=0.008239824324846268`, `aux_loss=0.013915486633777618`,
  `expert_usage=(0.37471845746040344, 0.38796138763427734,
  0.23732014000415802)`, `route_entropy=1.0765056610107422`,
  `update_count=2`, and checkpoint
  `/private/tmp/unilab-step21/live_moe_student.pt`.
- Live playback sentinel with that saved checkpoint: PASS with
  `policy_obs_mode=actor`, `action_dim=29`, `physics_shape=[1,72]`,
  `actions_shape=[1,29]`, `policy_checkpoint` equal to the saved MoE
  checkpoint, and `policy_action_nonzero=0.237480`.
- First playback attempt inside the sandbox failed at uv cache initialization:
  `failed to open file /Users/chengyuxuan/.cache/uv/sdists-v9/.git:
  Operation not permitted`; the same command passed when rerun with approved
  escalation for the `uv` cache boundary.

Status: COMPLETE for the shortest real-data distill-to-playback lifecycle.
Remaining unconfirmed boundaries are replay/storage integration, formal long
training, real trained teacher quality, trained student quality beyond a
one-step nonzero-action smoke check, and viewer-window validation.

### Step 3.12: Review And Gap Audit After Step 3.11

Scope: audit the generic G1 distillation migration after the live-dataset
MoE playback sentinel and choose the next safe implementation boundary.

Non-scope:

- no new code path;
- no replay/storage integration;
- no formal long training loop;
- no teacher or student policy quality claim;
- no viewer-window validation.

Files:

- No source code changes.
- Updated this migration note with the audit matrix and next-step decision.

Owner modules checked:

```text
conf/distill
 -> scripts/train_distill.py
 -> src/unilab/algos/torch/distill/{collector,data,teacher,trainer,offline,checkpoint,playback}.py
 -> scripts/deploy/check_unilab_g1_distill_{teacher_obs_contract,playback_live_sentinel}.py
 -> tests/{config,scripts,algos,visualization}
```

Coverage matrix:

| Layer | Files/symbols checked | Semantic aliases checked | Evidence | Gap |
| --- | --- | --- | --- | --- |
| Architecture/note | `note/g1_agile_height_distill_moe_migration.md`, `note/testing/test_inventory.md`, `note/testing/semantic_objects.md`, `note/architecture/architecture/01_unilab_repo_architecture.data.json` | default 99-D, legacy 100-D, checkpoint input dim, policy quality, replay/storage | note-confirmed | no new architecture module change in Step 3.12 |
| Config | `conf/distill/config.yaml`, `conf/distill/task/g1_walk_height/mujoco.yaml`, `tests/config/test_config_system.py` | `teacher.obs_dim`, `student.obs_dim`, `collect_teacher_projection`, MoE/offline overrides | contract-confirmed | no formal training-loop config |
| Entrypoint | `scripts/train_distill.py` | `collect_dataset_path`, `dry_run`, `offline_dataset_path`, `teacher.load_run`, `teacher.checkpoint` | code-confirmed and contract-confirmed | default fallthrough still raises `NotImplementedError`, so live behavior distillation training is intentionally not wired |
| Collector/data | `collector.py`, `data.py` | `student_obs`, `teacher_obs`, `teacher_projection`, `synthetic_teacher_tail`, dataset metadata | code-confirmed and contract-confirmed | collector only supports `action_mode=zero`; no replay or policy-action collector |
| Teacher/checkpoint | `teacher.py`, `checkpoint.py`, local `logs/fast_sac/*` probe | actor first rank-2 weight, `checkpoint_actor_input_dim`, normalizer state | code-confirmed, contract-confirmed, runtime-inspected | no local `logs/fast_sac/G1WalkHeight` teacher run found |
| Trainer/offline | `trainer.py`, `offline.py` | detached teacher action, behavior loss, aux loss, route entropy, expert usage, student checkpoint | code-confirmed and contract-confirmed | offline sequential tensor dataset only; no async runner/lifecycle integration |
| Playback | `playback.py`, `interactive_playback.py`, `check_unilab_g1_distill_playback_live_sentinel.py` | `LoadedDistillationStudentPolicy`, `policy_obs_mode=actor`, nonzero 29-D action | contract-confirmed and runtime-confirmed by Step 3.11 | no viewer-window validation and no quality claim |
| Stale search | `conf/`, `scripts/`, `src/`, `tests/`, `note/` | stale 100-D default, stale trained-quality/replay/formal-training claims | static-confirmed | remaining 100-D text is legacy/old-failure context, not active default |

Commands:

```bash
/opt/homebrew/bin/lean-ctx -c 'rg -n "Step 3\\.12|Immediate Next Step|replay/storage|formal training|teacher\\.obs_dim|collect_teacher_projection|trained policy quality|policy quality|99-D/100-D|100-D default" conf scripts src tests note -g "*.py" -g "*.yaml" -g "*.md" -g "*.json"'
/opt/homebrew/bin/lean-ctx -c 'find logs/fast_sac -maxdepth 3 -type f \( -name "model_*.pt" -o -name "run_config.json" -o -name "run_summary.json" \) | sort'
uv run python - <<'PY'
from pathlib import Path
import json
import torch

root = Path('logs/fast_sac')
summary = []
for task_dir in sorted(root.glob('*')):
    if not task_dir.is_dir():
        continue
    for run_dir in sorted(task_dir.glob('*')):
        ckpts = sorted(run_dir.glob('model_*.pt'))
        if not ckpts:
            continue
        latest = max(ckpts, key=lambda p: int(p.stem.split('_', 1)[1]) if '_' in p.stem and p.stem.split('_', 1)[1].isdigit() else -1)
        payload = torch.load(latest, map_location='cpu', weights_only=False)
        actor = payload.get('actor') if isinstance(payload, dict) else None
        actor_dim = None
        if isinstance(actor, dict):
            for key, value in actor.items():
                if 'weight' in str(key) and getattr(value, 'ndim', 0) == 2:
                    actor_dim = int(value.shape[1])
                    break
        summary.append((task_dir.name, run_dir.name, str(latest), actor_dim))
for item in summary:
    if item[0] in {'G1WalkHeight', 'G1WalkFlat', 'G1StandStill'}:
        print(item)
print({'g1_walk_height_runs': sum(1 for item in summary if item[0] == 'G1WalkHeight')})
PY
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill and not hora"
uv run pytest tests/visualization/test_interactive_playback.py -q -k distill
uv run pytest tests/scripts/test_visualization_entrypoints.py -q -k distill
uv run pytest tests/scripts/test_train_scripts.py -q -k distill_playback_live_sentinel
```

Evidence (2026-07-10):

- `scripts/train_distill.py` is still intentionally offline-phase only:
  the no-mode fallthrough raises `NotImplementedError` and points users to
  `training.collect_dataset_path`, `training.dry_run=true`, or
  `training.offline_dataset_path`.
- `collector.py` only supports `action_mode="zero"` and validates 99-D
  student/teacher tensor dimensions through owner projection helpers.
- `offline.py` consumes a saved tensor dataset sequentially and saves only a
  student checkpoint plus provenance metadata; it does not own replay storage
  or runner lifecycle.
- `trainer.py` keeps teacher actions under `torch.no_grad()` and reports
  behavior loss, aux loss, route entropy, and expert usage.
- Local checkpoint audit found no `logs/fast_sac/G1WalkHeight` runs:
  `{'g1_walk_height_runs': 0}`.
- Local `G1WalkFlat` SAC checkpoints are mixed 98-D/99-D/100-D and some later
  files lack `run_config.json`; they are not a confirmed height-conditioned
  teacher source.
- Targeted contract suite:
  - `uv run pytest tests/config/test_config_system.py -q -k distill`:
    PASS (`4 passed, 117 deselected`).
  - `uv run pytest tests/algos/test_g1_distillation_contract.py -q`:
    PASS (`20 passed`).
  - `uv run pytest tests/scripts/test_train_scripts.py -q -k "distill and not hora"`:
    PASS (`14 passed, 149 deselected`).
  - `uv run pytest tests/visualization/test_interactive_playback.py -q -k distill`:
    PASS (`5 passed, 12 deselected`).
  - `uv run pytest tests/scripts/test_visualization_entrypoints.py -q -k distill`:
    PASS (`2 passed, 15 deselected`).
  - `uv run pytest tests/scripts/test_train_scripts.py -q -k distill_playback_live_sentinel`:
    PASS (`3 passed, 160 deselected`).
- Fresh Step 21 rerun during this audit used
  `/private/tmp/unilab-step21-rerun`: live collection again produced
  `dataset_student_obs_dim=99`, `dataset_teacher_obs_dim=99`,
  `teacher_projection=identity`, and `synthetic_teacher_tail=False`; MoE
  offline update again produced `teacher_action_requires_grad=False`,
  `update_count=2`, non-empty `expert_usage`, and saved
  `/private/tmp/unilab-step21-rerun/live_moe_student.pt`; live playback
  sentinel loaded that checkpoint and passed with `policy_obs_mode=actor`,
  `actions_shape=[1,29]`, and `policy_action_nonzero=0.230114`.

Decision:

- Do not add a formal distillation training loop yet. The code path is already
  contract-confirmed for collect, offline update, checkpoint, and playback.
- The next blocking research/engineering input is a real teacher source:
  either train/select a 99-D `G1WalkHeight` SAC teacher checkpoint, or
  explicitly decide to use a legacy 100-D checkpoint with the documented
  `teacher.obs_dim=100` and `collect_teacher_projection=pad_zeros` bridge.
- The safest next implementation step is Step 3.13: teacher-run selection and
  preflight. That step should inspect candidate teacher checkpoints, validate
  actor input dim and run metadata, run the teacher-obs contract sentinel, and
  only then decide whether formal training-loop wiring is justified.

Status: COMPLETE for review-and-gap audit. The migration now has a closed
offline/playback contract, but not a real teacher/run selection or trained
policy-quality claim.

### Step 3.13: Teacher-Run Selection And Preflight

Scope: audit local SAC teacher checkpoint candidates for `G1WalkHeight`
distillation and decide whether any can be selected for formal behavior
distillation.

Non-scope:

- no teacher training;
- no formal long distillation loop;
- no new replay/storage integration;
- no policy quality claim;
- no implicit semantic upgrade of a `G1WalkFlat` checkpoint into a height
  teacher.

Candidate checkpoint metadata:

| Candidate | Metadata | Actor input dim | Preflight result | Decision |
| --- | --- | --- | --- | --- |
| `logs/fast_sac/G1WalkFlat/2026-06-27_03-01-36_mujoco/model_5000.pt` | `run_config.json` says `task_name=G1WalkFlat`; no height-command metadata found | 99 | PASS under default `teacher.obs_dim=99`, `collect_teacher_projection=identity`; dry-run update PASS | Technically compatible fallback only; not a confirmed `G1WalkHeight` teacher |
| `logs/fast_sac/G1WalkFlat/2026-07-08_15-42-50_mujoco/model_5000.pt` | no `run_config.json` | 100 | default teacher-obs sentinel FAILS; explicit `teacher.obs_dim=100`, `collect_teacher_projection=pad_zeros` dry-run PASS | Legacy bridge only; not selected as default teacher |
| `logs/fast_sac/G1WalkFlat/2026-07-09_02-48-58_mujoco/model_5000.pt` | no `run_config.json` | 98 | default dry-run FAILS with checkpoint actor input dim mismatch | Rejected |
| `logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt` | `run_config.json` says `task_name=G1StandStill` | 98 | default `g1_walk_height` teacher-obs sentinel FAILS with checkpoint actor input dim mismatch | Rejected for 99-D height teacher; reclassified in Step 3.19/3.20 as explicit 98-D standing teacher |

Commands:

```bash
uv run python - <<'PY'
from pathlib import Path
import json
import torch

candidates = [
    Path("logs/fast_sac/G1WalkFlat/2026-06-27_03-01-36_mujoco/model_5000.pt"),
    Path("logs/fast_sac/G1WalkFlat/2026-07-08_15-42-50_mujoco/model_5000.pt"),
    Path("logs/fast_sac/G1WalkFlat/2026-07-09_02-48-58_mujoco/model_5000.pt"),
    Path("logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt"),
]
for candidate in candidates:
    run_config = candidate.parent / "run_config.json"
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    actor = payload.get("actor", {}) if isinstance(payload, dict) else {}
    first_key = None
    actor_input_dim = None
    for key, value in actor.items():
        if "weight" in str(key) and getattr(value, "ndim", 0) == 2:
            first_key = key
            actor_input_dim = int(value.shape[1])
            break
    metadata = json.loads(run_config.read_text()) if run_config.exists() else {}
    print(candidate)
    print({
        "run_config_present": run_config.exists(),
        "task_name": metadata.get("task_name"),
        "algo_log_name": metadata.get("algo", {}).get("log_name"),
        "mode_observation": metadata.get("env", {}).get("mode_observation"),
        "observe_height_command": metadata.get("env", {}).get("observe_height_command"),
        "height_range": metadata.get("env", {}).get("height_range"),
        "first_weight_key": first_key,
        "actor_input_dim": actor_input_dim,
        "obs_normalizer_in_checkpoint": "obs_normalizer" in payload if isinstance(payload, dict) else False,
    })
PY
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py \
  --checkpoint-path logs/fast_sac/G1WalkFlat/2026-06-27_03-01-36_mujoco/model_5000.pt
uv run scripts/train_distill.py training.dry_run=true \
  teacher.load_run=logs/fast_sac/G1WalkFlat/2026-06-27_03-01-36_mujoco/model_5000.pt \
  teacher.checkpoint=-1 training.dry_run_batch_size=1 training.dry_run_updates=1 \
  'student.hidden_dims=[32]' algo.learning_rate=0.01 algo.max_grad_norm=10.0
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py \
  --checkpoint-path logs/fast_sac/G1WalkFlat/2026-07-08_15-42-50_mujoco/model_5000.pt
uv run scripts/train_distill.py training.dry_run=true \
  teacher.load_run=logs/fast_sac/G1WalkFlat/2026-07-08_15-42-50_mujoco/model_5000.pt \
  teacher.checkpoint=-1 teacher.obs_dim=100 training.collect_teacher_projection=pad_zeros \
  training.dry_run_batch_size=1 training.dry_run_updates=1 \
  'student.hidden_dims=[32]' algo.learning_rate=0.01 algo.max_grad_norm=10.0
uv run scripts/train_distill.py training.dry_run=true \
  teacher.load_run=logs/fast_sac/G1WalkFlat/2026-07-09_02-48-58_mujoco/model_5000.pt \
  teacher.checkpoint=-1 training.dry_run_batch_size=1 training.dry_run_updates=1 \
  'student.hidden_dims=[32]' algo.learning_rate=0.01 algo.max_grad_norm=10.0
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py \
  --checkpoint-path logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt
```

Evidence (2026-07-10):

- No local `logs/fast_sac/G1WalkHeight` run exists from the Step 3.12 audit.
- The 99-D `G1WalkFlat` checkpoint passes the default teacher-obs sentinel:
  live `G1WalkHeight` actor obs is `(1, 99)`, critic obs is `(1, 102)`,
  checkpoint input dim is `99`, and all sentinel checks pass.
- The same 99-D checkpoint also passes the default dry-run trainer update with
  `student_obs_shape=(1, 99)`, `teacher_obs_shape=(1, 99)`,
  `student_action_shape=(1, 29)`, `teacher_action_shape=(1, 29)`,
  `teacher_action_requires_grad=False`, and `update_count=1`.
- The 100-D checkpoint is rejected by the default 99-D route with
  `expected=99 got=100`, but passes only when explicitly configured as a legacy
  padded bridge.
- The 98-D `G1WalkFlat` checkpoint is rejected before trainer construction:
  `SAC teacher checkpoint obs dim mismatch: checkpoint actor input dim=98 ... configured teacher.obs_dim=99`.
- The 98-D `G1StandStill` checkpoint is rejected only by the default 99-D
  `G1WalkHeight` teacher-obs sentinel: `expected=99 got=98`. Step 3.19/3.20
  later reclassifies it as an explicit 98-D standing teacher.

Decision:

- No checkpoint is selected as a formal `G1WalkHeight` teacher.
- The 99-D `G1WalkFlat` checkpoint may be used only for plumbing or regression
  experiments if it is explicitly labeled as a semantic fallback.
- The 100-D checkpoint remains an explicit legacy bridge path, not the default.
- Formal distillation training should wait for a real 99-D `G1WalkHeight` SAC
  teacher checkpoint, or for an explicit research decision to accept the
  degraded `G1WalkFlat` fallback.

Status: COMPLETE for teacher-run selection and preflight. The blocker is now a
teacher-source decision, not an untested code path.

### Step 3.14: Teacher-Source Decision And Startup Preflight

Scope: choose the teacher-source route for `G1WalkHeight` behavior distillation
and prove the selected route can produce/load a 99-D SAC teacher checkpoint.

Non-scope:

- no long teacher training;
- no teacher quality claim;
- no formal distillation training loop;
- no use of `G1WalkFlat` as an implicit height teacher;
- no legacy 100-D bridge unless explicitly requested later.

Decision:

```text
Selected route: Option A, train or obtain a real 99-D `G1WalkHeight` SAC teacher.
Rejected as formal teacher: 99-D `G1WalkFlat` plumbing fallback.
Rejected as default route: 100-D legacy bridge.
Contract-only startup checkpoint:
  teacher.load_run=/private/tmp/unilab-step314-g1height-teacher-preflight-b2
  teacher.checkpoint=1
  teacher.obs_dim=99
  training.collect_teacher_projection=identity
```

Parameter inventory:

| param | owner | OFF / non-selected value | selected value | consumer | risk |
| --- | --- | --- | --- | --- | --- |
| `task` | `conf/offpolicy` / CLI | existing default `sac/g1_walk_flat/mujoco` | `sac/g1_walk_height/mujoco` | `scripts/train_offpolicy.py` | must activate height command/reward together |
| `training.task_name` | task owner YAML | `G1WalkFlat` | `G1WalkHeight` | log root, checkpoint metadata | wrong task name would hide teacher drift |
| `env.commands.observe_height_command` | `conf/offpolicy/task/sac/g1_walk_height/mujoco.yaml` | absent/false in flat task | `true` | actor obs layout | controls 99-D height actor obs |
| `env.commands.height_range` | height task YAML | absent | `[0.3, 0.754]` | command sampler, obs, reward | teacher must learn height-conditioned behavior |
| `teacher.load_run` | `conf/distill` override | `-1` or rejected fallback | selected G1WalkHeight run path | `scripts/train_distill.py` | formal distill must not auto-pick flat checkpoint |
| `teacher.checkpoint` | `conf/distill` override | `-1` | selected checkpoint id | teacher checkpoint resolver | must match selected run |
| `teacher.obs_dim` | `conf/distill` | `99` default | `99` | SAC teacher loader, sentinel | checkpoint actor input dim must match |
| `training.collect_teacher_projection` | `conf/distill` | `identity` default | `identity` | live dataset collector | no synthetic 100-D tail in selected route |

Execution contract:

```text
Owner module: `conf/offpolicy/task/sac/g1_walk_height/mujoco.yaml`,
`scripts/train_offpolicy.py`, `conf/distill`, and
`scripts/train_distill.py`.
Core parameter path:
  offpolicy height config
    -> 99-D SAC actor checkpoint
    -> teacher-obs preflight
    -> distill dry-run teacher load
Test class: live sentinel path for startup training plus checkpoint/load core
param path.
Stop condition: a `G1WalkHeight` SAC checkpoint exists, its first actor weight
has input dim 99, the teacher-obs sentinel passes, and `train_distill.py` can
load it with `teacher.load_run`, `teacher.checkpoint`, `teacher.obs_dim=99`,
and `identity` projection.
```

Commands:

```bash
uv run scripts/train_offpolicy.py task=sac/g1_walk_height/mujoco \
  algo.num_envs=1 algo.max_iterations=1 algo.save_interval=1 \
  algo.batch_size=1 algo.replay_buffer_n=8 algo.learning_starts=1 \
  algo.updates_per_step=1 algo.policy_frequency=1 \
  algo.algo_params.use_compile=false training.use_amp=false \
  training.no_play=true training.export_onnx=false \
  training.log_dir=/private/tmp/unilab-step314-g1height-teacher-preflight
uv run scripts/train_offpolicy.py task=sac/g1_walk_height/mujoco \
  algo.num_envs=1 algo.max_iterations=1 algo.save_interval=1 \
  algo.batch_size=2 algo.replay_buffer_n=8 algo.learning_starts=1 \
  algo.updates_per_step=1 algo.policy_frequency=1 \
  algo.algo_params.use_compile=false training.use_amp=false \
  training.no_play=true training.export_onnx=false \
  training.log_dir=/private/tmp/unilab-step314-g1height-teacher-preflight-b2
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py \
  --checkpoint-path /private/tmp/unilab-step314-g1height-teacher-preflight-b2/model_1.pt
uv run scripts/train_distill.py training.dry_run=true \
  teacher.load_run=/private/tmp/unilab-step314-g1height-teacher-preflight-b2 \
  teacher.checkpoint=1 training.dry_run_batch_size=1 \
  training.dry_run_updates=1 'student.hidden_dims=[32]' \
  algo.learning_rate=0.01 algo.max_grad_norm=10.0
```

Evidence (2026-07-10):

- The first startup command with `algo.batch_size=1` failed before training:
  `Symmetry augmentation requires batch_size divisible by 2, got 1`. This is a
  valid owner-contract rejection from the height SAC config where
  `algo.use_symmetry=true`.
- The second startup command with `algo.batch_size=2` completed one
  `G1WalkHeight` FastSAC iteration and saved
  `/private/tmp/unilab-step314-g1height-teacher-preflight-b2/model_1.pt`.
- The run summary reported `status=completed`, `total_env_steps=3`, and
  `last_checkpoint=/private/tmp/unilab-step314-g1height-teacher-preflight-b2/model_1.pt`.
- The saved run config has `training.task_name=G1WalkHeight`,
  `algo.algo=sac`, `algo.algo_log_name=fast_sac`, `algo.num_envs=1`,
  `algo.max_iterations=1`, `env.commands.observe_height_command=True`, and
  `env.commands.height_range=[0.3, 0.754]`.
- Teacher-obs sentinel on `model_1.pt` passed with live actor obs `(1, 99)`,
  live critic obs `(1, 102)`, checkpoint first weight `net.0.weight`, and
  checkpoint input dim `99`.
- Distill dry-run loaded the startup teacher with
  `teacher.load_run=/private/tmp/unilab-step314-g1height-teacher-preflight-b2`
  and `teacher.checkpoint=1`; it printed `student_obs_shape=(1, 99)`,
  `teacher_obs_shape=(1, 99)`, `student_action_shape=(1, 29)`,
  `teacher_action_shape=(1, 29)`, `teacher_action_requires_grad=False`,
  `student_grad_norm=0.7123597166970893`, and `update_count=1`.

Status: COMPLETE for teacher-source decision and startup preflight. Route A is
selected and executable. The startup checkpoint is contract/preflight evidence
only; it is not a trained teacher-quality artifact.

### Step 3.15: Bounded Real G1WalkHeight SAC Teacher Run

Scope: run a bounded real `G1WalkHeight` SAC teacher training/acquisition step
beyond the 1-iteration startup preflight, register the produced checkpoint, and
decide whether it is sufficient to justify formal distillation.

Non-scope:

- no student distillation training;
- no replay/storage migration;
- no viewer-window validation;
- no teacher quality claim from a short startup-quality run;
- no fallback to `G1WalkFlat` or 100-D legacy teacher routes.

Owner config group:

```text
flag name: offpolicy task override `task=sac/g1_walk_height/mujoco`
OFF behavior: default offpolicy paths remain unchanged and do not read distill
  teacher-source overrides.
ON behavior: `training.task_name=G1WalkHeight`,
  `env.commands.observe_height_command=True`, `height_range=[0.3, 0.754]`,
  `algo.algo=sac`, `algo.algo_log_name=fast_sac`, and checkpoints are written
  under the selected `G1WalkHeight` run path.
forbidden mixed states: a `G1WalkFlat`, 98-D, or legacy 100-D checkpoint is not
  promoted to the formal teacher source.
tests proving OFF: targeted offpolicy config tests still pass.
tests proving ON: bounded live SAC run, teacher-obs checkpoint sentinel, and
  distill dry-run teacher load.
```

Parameter inventory:

| param | owner | old default | Step 3.15 value | consumer | persistence/playback risk |
| --- | --- | --- | --- | --- | --- |
| `task` | `conf/offpolicy/config.yaml` + CLI | `sac/g1_walk_flat/mujoco` | `sac/g1_walk_height/mujoco` | `scripts/train_offpolicy.py` | selects height obs/reward owner |
| `algo.num_envs` | `conf/offpolicy/algo/sac.yaml` + CLI | `4096` | `4` | FastSAC runner | bounded run only, not quality training |
| `algo.max_iterations` | SAC config + CLI | `500` | `8` | FastSAC learner loop | produces checkpoint but weak quality |
| `algo.batch_size` | SAC config + CLI | `8192` | `8` | learner batch | must remain even because symmetry is enabled |
| `training.log_dir` | `scripts/train_offpolicy.py` | timestamped log root | `logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco` | checkpoint resolver, distill teacher load | selected teacher run path |
| `teacher.load_run` | `conf/distill` override | `-1` | `logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco` | `scripts/train_distill.py` | formal distill should use this only after quality gate |
| `teacher.checkpoint` | `conf/distill` override | `-1` | `8` | teacher loader | must match `model_8.pt` |
| `teacher.obs_dim` | `conf/distill` | `99` | `99` | SAC teacher preflight | checkpoint actor input dim must match |
| `training.collect_teacher_projection` | `conf/distill` | `identity` | `identity` | live dataset collector | no synthetic 100-D tail |

Execution contract:

```text
Owner module:
  conf/offpolicy/task/sac/g1_walk_height/mujoco.yaml
  -> scripts/train_offpolicy.py
  -> logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco
  -> scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py
  -> scripts/train_distill.py
Core parameter path:
  task=sac/g1_walk_height/mujoco
    -> 99-D SAC actor checkpoint model_8.pt
    -> checkpoint actor input dim preflight
    -> distill teacher load with teacher.checkpoint=8
    -> detached teacher action shape (B,29)
Test class: live sentinel path plus checkpoint/load core parameter path.
Stop condition: a bounded `G1WalkHeight` checkpoint exists, loads as a 99-D SAC
teacher, and the run-quality summary says whether to proceed or stop.
```

Commands:

```bash
uv run scripts/train_offpolicy.py task=sac/g1_walk_height/mujoco \
  algo.num_envs=4 algo.max_iterations=8 algo.save_interval=4 \
  algo.batch_size=8 algo.replay_buffer_n=64 algo.learning_starts=1 \
  algo.updates_per_step=1 algo.policy_frequency=1 \
  algo.algo_params.use_compile=false training.use_amp=false \
  training.no_play=true training.export_onnx=false \
  training.log_dir=logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py \
  --checkpoint-path logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco/model_8.pt
uv run scripts/train_distill.py training.dry_run=true \
  teacher.load_run=logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco \
  teacher.checkpoint=8 training.dry_run_batch_size=2 \
  training.dry_run_updates=1 'student.hidden_dims=[32]' \
  algo.learning_rate=0.01 algo.max_grad_norm=10.0
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/config/test_config_system.py -q -k \
  "offpolicy_g1_walk_flat_mujoco_td3_uses_td3_task_owner or offpolicy_g1_walk_flat_motrix_sac_preserves_backend_overrides"
```

Evidence (2026-07-10):

- Bounded live teacher run completed: `status=completed`,
  `completed_iterations=8`, `total_env_steps=40`, wall time about 15.38 sec,
  and `last_checkpoint=logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco/model_8.pt`.
- Runtime training panel reported `FastSAC | G1WalkHeight | iter 8/8`,
  `Buffer=40`, `Timeout Rate=0.0%`, `Terminated Rate=0.0%`,
  `Action Std=0.0979`, `Policy Entropy=-26.304`, and
  `Reward Mean 0.000 / Peak 0.000`.
- Saved run config confirms `training.task_name=G1WalkHeight`,
  `algo.algo=sac`, `algo.algo_log_name=fast_sac`, `algo.num_envs=4`,
  `algo.max_iterations=8`, `algo.save_interval=4`,
  `env.commands.observe_height_command=True`, and
  `env.commands.height_range=[0.3, 0.754]`.
- Teacher-obs sentinel on `model_8.pt`: PASS with live actor obs `(1, 99)`,
  critic obs `(1, 102)`, checkpoint first weight `net.0.weight`, and
  checkpoint input dim `99`.
- Distill dry-run with `teacher.load_run` set to the bounded run and
  `teacher.checkpoint=8`: PASS with `student_obs_shape=(2, 99)`,
  `teacher_obs_shape=(2, 99)`, `student_action_shape=(2, 29)`,
  `teacher_action_shape=(2, 29)`, `teacher_action_requires_grad=False`,
  `student_grad_norm=0.4746024409423377`, `loss=0.07399463653564453`, and
  `update_count=1`.
- OFF/config impact checks:
  - `uv run pytest tests/config/test_config_system.py -q -k distill`: PASS
    (`4 passed, 117 deselected`).
  - `uv run pytest tests/config/test_config_system.py -q -k "offpolicy_g1_walk_flat_mujoco_td3_uses_td3_task_owner or offpolicy_g1_walk_flat_motrix_sac_preserves_backend_overrides"`:
    PASS (`2 passed, 119 deselected`).

Decision:

- `model_8.pt` is selected as a registered bounded `G1WalkHeight` SAC teacher
  checkpoint for contract/load testing:
  `teacher.load_run=logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco`,
  `teacher.checkpoint=8`, `teacher.obs_dim=99`,
  `training.collect_teacher_projection=identity`.
- `model_8.pt` is not accepted as a quality teacher for formal student
  training. The bounded run is too short and the reward summary is still
  `Reward Mean 0.000 / Peak 0.000`, with `final_mean_reward=None` and
  `best_mean_reward=None` in `run_summary.json`.
- Do not proceed to formal distillation quality claims from this checkpoint.
  It can be used only for route/regression preflight unless a later teacher
  quality gate accepts a longer run.

Status: COMPLETE for bounded teacher training/acquisition and checkpoint
preflight. The remaining blocker is teacher quality, not teacher source,
checkpoint dimension, or distill loader compatibility.

### Step 3.16: Teacher-Quality Gate

Scope: decide whether the bounded `G1WalkHeight` SAC checkpoint from Step 3.15
is accepted for formal student distillation, requires more teacher training, or
is kept only as route-regression plumbing.

Non-scope:

- no student distillation training;
- no collector/action-source expansion;
- no viewer-window validation;
- no new teacher architecture or reward tuning;
- no automatic fallback to `G1WalkFlat` or legacy 100-D checkpoints.

Quality gate:

```text
Required for formal student distillation:
1. checkpoint contract: 99-D actor input, live 99-D `G1WalkHeight` obs, and
   `training.collect_teacher_projection=identity`;
2. run lifecycle: completed run with a selected checkpoint and no load/play
   crash;
3. minimal behavior signal: policy rollout must be finite, avoid immediate
   termination/truncation, and outperform a zero-action baseline on the same
   short probe before it can be used as a teacher-quality source.

Route-regression only:
  checkpoints that satisfy 1 and 2 but fail 3.
```

Owner modules:

```text
logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco/run_summary.json
 -> scripts/train_offpolicy.py load/play actor construction
 -> G1WalkHeight live env reset/step
 -> policy-vs-zero short rollout quality probe
 -> distill teacher selection decision
```

Core parameter path:

```text
model_8.pt
  -> SAC actor load
  -> live obs shape (1,99)
  -> deterministic teacher action shape (1,29)
  -> env.step reward/terminated/truncated facts
  -> compare policy reward mean with zero-action reward mean
```

Commands:

```bash
uv run python /private/tmp/unilab_step316_teacher_quality_probe.py
```

The temporary probe script builds the same `G1WalkHeight` offpolicy task config,
loads `model_8.pt`, runs 16 policy-action steps, resets, then runs 16
zero-action steps, and prints reward/action/termination/finite facts. It is a
runtime diagnostic artifact under `/private/tmp`, not a repo source file.

Evidence (2026-07-10):

- The first module-local probe attempt failed because a `/private/tmp` script
  does not automatically put the repo root on `sys.path`; adding `os.getcwd()`
  to `sys.path` fixed only the temporary probe.
- The second probe attempt found the correct owner import:
  `create_env` and `ensure_registries` are re-exported from `unilab.training`,
  not `unilab.base.registry`; this again changed only the temporary probe.
- Policy rollout with `model_8.pt`: `obs_groups_spec={'obs': 99, 'critic': 102}`,
  `obs_dim=99`, `action_dim=29`, `steps=16`, `reward_mean=0.30881981179118156`,
  `reward_min=0.26820215582847595`, `reward_max=0.3504319190979004`,
  `action_abs_max_mean=0.13940076529979706`,
  `action_abs_max_max=0.14286206662654877`, `terminated_total=0`,
  `truncated_total=0`, `finite_obs_all=True`, and `finite_actions_all=True`.
- Zero-action baseline on the same probe: `steps=16`,
  `reward_mean=0.3380646537989378`, `reward_min=0.31531044840812683`,
  `reward_max=0.3643770217895508`, `action_abs_max_mean=0.0`,
  `terminated_total=0`, `truncated_total=0`, `finite_obs_all=True`, and
  `finite_actions_all=True`.
- Differential result: `policy_minus_zero_reward_mean=-0.029244842007756233`.

Decision:

- `model_8.pt` fails the teacher-quality gate because it does not outperform
  the zero-action baseline on the short live rollout probe.
- Keep `model_8.pt` only as a route-regression / checkpoint-load / dimension
  preflight artifact:
  `teacher.load_run=logs/fast_sac/G1WalkHeight/2026-07-10_step315_bounded_mujoco`,
  `teacher.checkpoint=8`, `teacher.obs_dim=99`,
  `training.collect_teacher_projection=identity`.
- Do not use `model_8.pt` as the formal behavior teacher for student training.
- The next productive path is either to continue teacher training with a real
  iteration budget until the teacher beats zero-action and has meaningful
  reward/episode summaries, or to acquire a stronger existing 99-D
  `G1WalkHeight` SAC teacher checkpoint.

Status: COMPLETE for teacher-quality gate. Teacher source and checkpoint
compatibility are solved, but teacher quality is not accepted.

### Step 3.17: Collection Action-Source Contract

Scope: return to the original migration plan's action-source contract and add
one explicit non-zero action source for live distillation dataset collection.

Non-scope:

- no teacher-policy collection;
- no student-policy collection;
- no replay/action-replay collector;
- no formal student distillation training;
- no teacher-quality acceptance for `model_8.pt`.

Decision:

```text
Selected action-source step: add `training.collect_action_mode=random`.
Default OFF path: `training.collect_action_mode=zero`, unchanged.
ON path: `training.collect_action_mode=random` with optional
`training.collect_action_seed`.
Reason: this removes the weakest collection-path gap first: live collection is
no longer restricted to zero actions. It is still a distribution/connectivity
probe, not a teacher-quality or policy-quality claim.
```

Parameter inventory:

| param | owner | old default | new default | OFF value | ON value | consumers | persistence/playback risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `training.collect_action_mode` | `conf/distill/config.yaml` | `zero` | `zero` | zero actions | `random` random actions | `scripts/train_distill.py`, `collector.py` | saved dataset metadata records action source |
| `training.collect_action_seed` | `conf/distill/config.yaml` | absent | `null` | inert | integer seed | `collector.py` random generator | reproducibility only; no playback effect |
| `action_abs_max` | dataset metadata | absent | recorded | `0.0` for zero | positive for random | saved `DistillationTensorDataset` metadata | diagnostic only |

Feature-flag contract:

```text
flag name: `training.collect_action_mode`
OFF behavior: `zero` keeps previous live collection behavior exactly: zero
  actions, saved metadata `action_abs_max=0.0`.
ON behavior: `random` samples finite uniform actions in [-1, 1], steps the live
  env, and records `action_seed` plus `action_abs_max`.
forbidden mixed states: unsupported action modes fail closed with
  `Unsupported collect action_mode`.
tests proving OFF: config and script collection tests keep default zero/null.
tests proving ON: collector fake-env test, script fake-env test, and live MuJoCo
  random collection sentinel.
```

Files:

- Modified: `conf/distill/config.yaml`
  - Adds inert `training.collect_action_seed: null`.
- Modified: `src/unilab/algos/torch/distill/collector.py`
  - Supports `action_mode in {"zero", "random"}`.
  - Generates finite random actions with optional seed.
  - Records `action_seed` and `action_abs_max` in dataset metadata.
- Modified: `scripts/train_distill.py`
  - Passes `training.collect_action_seed` into the collector.
  - Reports `collect_action_seed` and `collect_action_abs_max`.
- Modified: tests and testing notes.

Core parameter path:

```text
training.collect_action_mode=random
  -> scripts/train_distill.py
  -> collect_distillation_dataset_from_env(..., action_mode="random")
  -> finite random actions shape (num_envs, action_dim)
  -> env.step(actions)
  -> DistillationTensorDataset.metadata[action_abs_max > 0]
  -> saved dataset reload
```

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "collect_distillation_dataset_from_env"
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill_script_collects"
uv run pytest tests/config/test_config_system.py -q -k distill
uv run scripts/train_distill.py \
  training.collect_dataset_path=/private/tmp/unilab-step317-random-live-dataset.pt \
  training.collect_num_samples=3 training.collect_num_envs=1 \
  training.collect_action_mode=random training.collect_action_seed=11
uv run python -c "from unilab.algos.torch.distill import load_distillation_dataset; d=load_distillation_dataset('/private/tmp/unilab-step317-random-live-dataset.pt', expected_student_obs_dim=99, expected_teacher_obs_dim=99); print({'num_samples': d.num_samples, 'student_obs_dim': d.student_obs_dim, 'teacher_obs_dim': d.teacher_obs_dim, 'metadata': d.metadata})"
```

Evidence (2026-07-10):

- Test-first failure before implementation:
  `TypeError: collect_distillation_dataset_from_env() got an unexpected keyword
  argument 'action_seed'`.
- Collector contract tests: PASS (`4 passed, 17 deselected`).
- Script collection tests: PASS (`2 passed, 162 deselected`).
- Distill config tests: PASS (`4 passed, 117 deselected`).
- Live MuJoCo random collection sentinel: PASS with `dataset_num_samples=3`,
  `dataset_student_obs_dim=99`, `dataset_teacher_obs_dim=99`,
  `action_mode=random`, `action_seed=11`, `action_abs_max=0.9928717613220215`,
  `num_envs=1`, `env_steps=2`, `teacher_projection=identity`, and
  `synthetic_teacher_tail=False`.
- Saved dataset reload: PASS with the same 99/99 dims and random-action metadata.

Status: COMPLETE for the first non-zero collection action source. Remaining
action-source gaps are teacher-policy collection, student-policy collection,
and replay/action-replay collection. Teacher quality remains unaccepted.

### Step 3.18: Current Migration Diff Stabilization

Scope: freeze the current migration surface, run a focused impact suite, and
separate real blockers from already-covered plumbing.

Non-scope:

- no new distillation feature;
- no new teacher checkpoint acceptance;
- no formal student training loop;
- no viewer-window claim;
- no branch switch.

Current migration surface:

- Tracked modifications:
  - `note/architecture/architecture/01_unilab_repo_architecture.data.json`
  - `note/g1_agile_height_distill_moe_migration.md`
  - `note/testing/semantic_objects.md`
  - `note/testing/test_inventory.md`
  - `scripts/play_interactive.py`
  - `src/unilab/visualization/interactive_playback.py`
  - `tests/config/test_config_system.py`
  - `tests/scripts/test_train_scripts.py`
  - `tests/scripts/test_visualization_entrypoints.py`
  - `tests/visualization/test_interactive_playback.py`
- Untracked migration additions:
  - `conf/distill/`
  - `scripts/train_distill.py`
  - `scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py`
  - `scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py`
  - `src/unilab/algos/torch/distill/`
  - `tests/algos/test_g1_distillation_contract.py`

Test classification:

- `conf/distill` and script entrypoints: secondary contract path.
- `collector.py` action source and dataset metadata: core param path.
- `scripts/train_distill.py ... collect_action_mode=random`: live sentinel path.
- architecture JSON and stale search: review checkpoint path.

Commands and evidence (2026-07-10):

```bash
uv run ruff check src/unilab/algos/torch/distill scripts/train_distill.py scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py tests/visualization/test_interactive_playback.py tests/scripts/test_visualization_entrypoints.py
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run pytest tests/visualization/test_interactive_playback.py -q -k distill
uv run pytest tests/scripts/test_visualization_entrypoints.py -q -k distill
uv run python -m py_compile scripts/train_distill.py scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py
uv run python -c "import json; from pathlib import Path; paths=list(Path('note/architecture').glob('**/*.data.json')); [json.loads(p.read_text()) for p in paths]; print({'json_files': len(paths)})"
/opt/homebrew/bin/lean-ctx -c 'git diff --check'
/opt/homebrew/bin/lean-ctx -c 'rg -n "TODO|placeholder|pass$|NotImplementedError" conf/distill scripts/train_distill.py scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py src/unilab/algos/torch/distill tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py tests/visualization/test_interactive_playback.py tests/scripts/test_visualization_entrypoints.py'
uv run scripts/train_distill.py training.collect_dataset_path=/private/tmp/unilab-step318-random-live-dataset.pt training.collect_num_samples=3 training.collect_num_envs=1 training.collect_action_mode=random training.collect_action_seed=11
```

Results:

- Ruff: PASS, `All checks passed!`.
- Distill config contract: PASS, `4 passed, 117 deselected`.
- Distillation algorithm contracts: PASS, `21 passed`.
- Script distillation impact tests: PASS, `31 passed, 133 deselected`.
- Distill playback visualization tests: PASS, `5 passed, 12 deselected`.
- Visualization entrypoint tests: PASS, `2 passed, 15 deselected`.
- Py compile for distill entrypoints: PASS.
- Architecture JSON parse: PASS, `{'json_files': 4}`.
- `git diff --check`: PASS.
- Stale/placeholder scan:
  - no `TODO` or `placeholder` in the migration implementation;
  - test placeholders are confined to existing fixture bytes;
  - `scripts/train_distill.py` keeps one intentional `NotImplementedError` as a
    fail-closed guard for the not-yet-wired live training loop.
- Live MuJoCo random collection sentinel: PASS with
  `dataset_num_samples=3`, `dataset_student_obs_dim=99`,
  `dataset_teacher_obs_dim=99`, `action_mode=random`, `action_seed=11`,
  `action_abs_max=0.9928717613220215`, `num_envs=1`, `env_steps=2`,
  `teacher_projection=identity`, and `synthetic_teacher_tail=False`.

Review decision:

- The current migration diff is stable enough as plumbing and contract
  infrastructure.
- The random-action dataset collection path is runtime-confirmed, but it is
  still a distribution/connectivity probe, not a teacher-quality or
  policy-quality claim.
- The formal live distillation training loop remains intentionally fail-closed
  unless the user selects a teacher checkpoint and an explicit training path.
- The main unresolved blocker is still teacher quality, not current plumbing.

Status: COMPLETE for Step 3.18 focused impact stabilization.

### Step 3.19: Reclassify G1StandStill Checkpoint as Standing Teacher

Scope: evaluate the user-selected checkpoint
`logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt` as the
teacher source for the next distillation step.

Non-scope:

- no code change yet;
- no formal student distillation;
- no claim that a standing teacher covers walking or height tracking;
- no silent 99-D to 98-D truncation.

Checkpoint facts:

- Run identity: `algo=sac`, `task=G1StandStill`, `sim_backend=mujoco`.
- Training summary: `completed_iterations=5000`,
  `total_env_steps=10262528`, `final_mean_reward=-331.5979050613403`,
  `mean_episode_length=731.15`.
- Actor checkpoint input: `net.0.weight` has shape `(512, 98)`.
- G1 env obs contract:
  - `G1StandStill`: obs `98`, critic `101`.
  - `G1WalkHeight`: obs `99`, critic `102`.
  - The one-dim difference is explained by `G1WalkHeight`
    `commands.observe_height_command=true`, which appends one height-command
    column to `command_obs`.

Runtime evidence (2026-07-10):

```bash
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py \
  --checkpoint-path logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt
```

Result under the current default `distill/task=g1_walk_height/mujoco`:

- PASS: live student/env obs `99`.
- PASS: live reset shapes `obs=(1, 99)`, `critic=(1, 102)`.
- PASS: configured teacher/live dim `99` under identity projection.
- FAIL: checkpoint input dim mismatch, expected `99`, got `98`.

Teacher-load contract:

```bash
uv run python -c "... DistillationTeacherSpec(obs_dim=98, action_dim=29) ..."
```

Result:

- PASS: teacher loads with `obs_dim=98`, `action_dim=29`.
- PASS: frozen teacher action shape `(2, 29)`.
- PASS: `requires_grad=False`, finite action, `abs_max=0.13766266405582428`.

Negative control:

```bash
uv run python -c "... DistillationTeacherSpec(obs_dim=99, action_dim=29) ..."
```

Result:

- FAIL as expected with `SAC teacher checkpoint obs dim mismatch:
  checkpoint actor input dim=98 ..., configured teacher.obs_dim=99`.

Standing-quality sentinel:

```bash
uv run scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py \
  --num-envs 4 --steps 64 --probe-mode support-policy-diff
```

Process status is FAIL because the sentinel includes the `zero_action` control
group in global threshold checks. The role-level facts still support using the
checkpoint as a standing teacher candidate:

- `policy_actor.checkpoint_path` resolves to the selected `model_5000.pt`.
- `obs_groups_spec={"obs": 98, "critic": 101}`.
- `current_trained_policy_action` at final step:
  - `terminated_total=0`;
  - `tilt_deg=0.6464956402778625`;
  - `base_height=0.7348198294639587`;
  - `both_feet_contact=1.0`;
  - `contact_balance=0.0`;
  - `reward_mean=0.26723846793174744`.
- `zero_action` control at final step:
  - `tilt_deg` range approximately `35.50` to `41.91`;
  - `base_height` range approximately `0.5809` to `0.5967`;
  - `both_feet_contact=0.0`;
  - `reward_mean` approximately `-3.71`.

Decision:

- Accept this checkpoint as a `G1StandStill` standing-teacher candidate.
- Do not use it as the default `G1WalkHeight` teacher.
- Do not add a generic 99-D to 98-D teacher projection yet. The missing dim is
  specifically the height-command observation, so the safer integration is an
  explicit `distill/task=g1_stand_still/mujoco` config with
  `teacher.obs_dim=98`, `student.obs_dim=98`, `training.task_name=G1StandStill`,
  and no height-command observation.
- If the later MoE design needs standing + walking experts, this checkpoint can
  become the standing expert source, while a separate 99-D height/walk teacher
  remains necessary for height-conditioned walking.

Status: ACCEPTED as standing-teacher candidate only. Not accepted as
height-conditioned walking teacher.

### Step 3.20: Explicit G1StandStill Distill Owner Config

Scope: add an explicit distillation task owner for the accepted 98-D standing
teacher route.

Non-scope:

- no change to the default `distill/task=g1_walk_height/mujoco` route;
- no implicit 99-D to 98-D projection;
- no formal long training loop;
- no claim that the standing teacher solves height-conditioned walking.

Files:

- Created: `conf/distill/task/g1_stand_still/mujoco.yaml`
  - Owns `training.task_name=G1StandStill`.
  - Owns zero-command standing env/reward parameters.
  - Explicitly disables height-command observation.
  - Selects the accepted standing teacher run:
    `teacher.load_run=2026-07-09_22-55-05_mujoco`,
    `teacher.checkpoint=5000`.
  - Sets `teacher.obs_dim=98` and `student.obs_dim=98`.
- Modified: `tests/config/test_config_system.py`
  - Adds the config-level owner contract for `task=g1_stand_still/mujoco`.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds 98-D stand-still teacher/student assembly coverage.
  - Adds 98-D teacher obs checkpoint preflight coverage.
  - Adds 98/98 dataset collection coverage through the stand-still owner config.

Feature-flag contract:

```text
flag name: Hydra owner group `task=g1_stand_still/mujoco`
OFF behavior: default `conf/distill/config.yaml` still selects
  `task=g1_walk_height/mujoco`, `teacher.obs_dim=99`, `student.obs_dim=99`,
  and height-command observation enabled.
ON behavior: `task=g1_stand_still/mujoco` selects `G1StandStill`,
  `teacher.obs_dim=98`, `student.obs_dim=98`, zero commands, no height-command
  observation, and the accepted `model_5000.pt` standing teacher.
generated/derived overrides: `teacher.load_run`, `teacher.checkpoint`,
  `teacher.task_name`, `teacher.task`, `student.obs_dim`, `env.commands.*`.
forbidden mixed states: do not route the 98-D standing checkpoint through
  `g1_walk_height`; do not silently truncate the 99-D height obs.
tests proving OFF: `test_distill_g1_walk_height_owner_composes`.
tests proving ON: `test_distill_g1_stand_still_owner_composes`,
  `test_distill_script_builds_stand_still_teacher_and_student_from_owner_config`,
  `test_g1_distill_teacher_obs_contract_reports_stand_still_identity`,
  `test_distill_script_collects_stand_still_dataset_with_owner_config`.
```

Commands and evidence (2026-07-10):

```bash
uv run pytest tests/config/test_config_system.py -q -k "distill_g1_stand_still or distill_g1_walk_height"
uv run pytest tests/scripts/test_train_scripts.py -q -k "stand_still and distill"
uv run ruff check conf/distill tests/config/test_config_system.py tests/scripts/test_train_scripts.py
uv run scripts/train_distill.py task=g1_stand_still/mujoco training.dry_run=true training.dry_run_batch_size=2 training.dry_run_updates=1 teacher.load_run=2026-07-09_22-55-05_mujoco teacher.checkpoint=5000
uv run scripts/train_distill.py task=g1_stand_still/mujoco training.dry_run=true training.dry_run_batch_size=2 training.dry_run_updates=1
uv run scripts/train_distill.py task=g1_stand_still/mujoco training.collect_dataset_path=/private/tmp/unilab-step320-stand-still-dataset.pt training.collect_num_samples=3 training.collect_num_envs=1 training.collect_action_mode=random training.collect_action_seed=11
uv run python -c "from unilab.algos.torch.distill import load_distillation_dataset; d=load_distillation_dataset('/private/tmp/unilab-step320-stand-still-dataset.pt', expected_student_obs_dim=98, expected_teacher_obs_dim=98); print({'num_samples': d.num_samples, 'student_obs_dim': d.student_obs_dim, 'teacher_obs_dim': d.teacher_obs_dim, 'metadata': d.metadata})"
```

Results:

- Config owner tests: PASS, `2 passed, 120 deselected`.
- Script owner tests: PASS, `3 passed, 164 deselected`.
- Ruff on changed config/test files: PASS.
- Real checkpoint dry-run: PASS with `student_obs_shape=(2, 98)`,
  `teacher_obs_shape=(2, 98)`, `teacher_action_shape=(2, 29)`,
  `teacher_action_requires_grad=False`, `update_count=1`, and finite loss.
- Owner-config dry-run without teacher overrides: PASS with the same 98/98
  teacher/student shapes, proving `teacher.load_run` and `teacher.checkpoint`
  are selected by `task=g1_stand_still/mujoco` itself.
- Real live collection: PASS with `dataset_num_samples=3`,
  `dataset_student_obs_dim=98`, `dataset_teacher_obs_dim=98`,
  `action_mode=random`, `action_seed=11`,
  `action_abs_max=0.9928717613220215`, `env_steps=2`,
  `teacher_projection=identity`, and `synthetic_teacher_tail=False`.
- Saved stand-still dataset reload: PASS with `num_samples=3`,
  `student_obs_dim=98`, `teacher_obs_dim=98`, and stand-still metadata.

Status: COMPLETE for the explicit stand-still distill owner config.

### Step 3.21: Stand-Still Teacher-Policy Collection and Saved-Dataset Update

Scope: wire a real teacher-policy action source only under the accepted
`g1_stand_still/mujoco` 98-D owner route, then run a bounded offline student
update from the saved live dataset.

Non-scope:

- no change to default `g1_walk_height/mujoco`;
- no use of the 98-D standing checkpoint as a 99-D height-conditioned teacher;
- no replay buffer, async collector, or formal long training lifecycle;
- no policy-quality claim beyond finite short-chain execution.

Files:

- Modified: `src/unilab/algos/torch/distill/collector.py`
  - Adds `action_mode=teacher_policy`.
  - Requires an explicit frozen teacher policy argument.
  - Converts projected teacher obs to detached teacher actions and checks
    rank, shape, finiteness, and `action_abs_max`.
- Modified: `scripts/train_distill.py`
  - Resolves/loads the SAC teacher before live collection only when
    `training.collect_action_mode=teacher_policy`.
  - Rejects teacher-policy collection unless the route is exactly
    `G1StandStill`, 98-D teacher/student obs, identity projections, `obs`
    teacher key, no height-command observation, and no action seed.
  - Writes `teacher_policy_checkpoint_path` into dataset metadata and probe
    output.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds collector-level teacher-policy action-source coverage.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds default height-route rejection coverage.
  - Adds stand-still 98-D teacher-policy collection plus bounded saved-dataset
    offline update coverage.

Feature-flag contract:

```text
flag name: training.collect_action_mode=teacher_policy
OFF behavior: default collection stays zero-action; random collection remains
  an explicit connectivity probe.
ON behavior: only `task=g1_stand_still/mujoco` may load the accepted 98-D
  standing SAC teacher and use teacher actions to step the live env.
forbidden mixed states: `task=g1_walk_height/mujoco` + `teacher_policy`
  raises before env creation; height-command observation, non-identity
  projection, or 99-D/98-D mixed obs also raise.
dataset proof: saved metadata includes `action_mode=teacher_policy`,
  `teacher_policy_checkpoint_path`, `env_steps>0`, `action_abs_max>0`,
  `teacher_projection=identity`, and `synthetic_teacher_tail=False`.
```

Commands and evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "teacher_policy or collect_distillation_dataset_from_env"
uv run pytest tests/scripts/test_train_scripts.py -q -k "teacher_policy or stand_still"
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill and not hora"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run ruff check src/unilab/algos/torch/distill scripts/train_distill.py scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py
uv run scripts/train_distill.py task=g1_stand_still/mujoco training.collect_dataset_path=/private/tmp/unilab-step321-stand-teacher-policy-dataset.pt training.collect_num_samples=3 training.collect_num_envs=1 training.collect_action_mode=teacher_policy
uv run scripts/train_distill.py task=g1_stand_still/mujoco training.offline_dataset_path=/private/tmp/unilab-step321-stand-teacher-policy-dataset.pt training.offline_batch_size=2 training.offline_max_updates=1 training.offline_checkpoint=/private/tmp/unilab-step321-stand-student.pt
uv run python -c "from unilab.algos.torch.distill import load_distillation_student_policy; p=load_distillation_student_policy('/private/tmp/unilab-step321-stand-student.pt', device='cpu'); print({'obs_dim': p.obs_dim, 'action_dim': p.action_dim, 'student_model_type': p.distill_runtime_cfg.get('student_model_type'), 'distill_source': p.distill_runtime_cfg.get('distill_source'), 'dataset_path': p.distill_runtime_cfg.get('dataset_path')})"
```

Results:

- Collector teacher-policy tests: PASS, `5 passed, 17 deselected`.
- Script stand-still/teacher-policy tests: PASS, `5 passed, 164 deselected`.
- Config distill tests: PASS, `5 passed, 117 deselected`.
- Script distill focused suite: PASS, `20 passed, 149 deselected`.
- Distill algos contract suite: PASS, `22 passed`.
- Ruff focused suite: PASS.
- Real stand-still teacher-policy collection: PASS with
  `dataset_num_samples=3`, `dataset_student_obs_dim=98`,
  `dataset_teacher_obs_dim=98`, `env_steps=2`,
  `action_abs_max=0.09451983124017715`, `action_mode=teacher_policy`,
  `teacher_projection=identity`, `student_projection=identity`,
  `synthetic_teacher_tail=False`, and
  `teacher_policy_checkpoint_path=logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt`.
- Saved-dataset offline update: PASS with `student_obs_shape=(2, 98)`,
  `teacher_obs_shape=(2, 98)`, `teacher_action_shape=(2, 29)`,
  `teacher_action_requires_grad=False`, `update_count=1`, finite loss, and
  checkpoint `/private/tmp/unilab-step321-stand-student.pt`.
- Student checkpoint reload: PASS with `obs_dim=98`, `action_dim=29`,
  `student_model_type=mlp`, and `distill_source=offline_dataset`.

Status: COMPLETE for bounded 98-D stand-still teacher-policy collection and
saved-dataset offline student update.

### Step 3.22: Stand-Still Student Live Playback Sentinel

Scope: prove the 98-D student checkpoint produced by Step 3.21 can enter the
real MuJoCo distill playback path under the `g1_stand_still/mujoco` owner route
and emit finite nonzero policy actions.

Non-scope:

- no viewer-window validation;
- no policy-quality or standing-stability claim;
- no change to default `g1_walk_height/mujoco` playback;
- no formal long training lifecycle.

Files:

- Modified: `scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py`
  - Adds `--task`, defaulting to `g1_walk_height/mujoco`.
  - Uses the selected distill owner config when composing Hydra.
  - Reports owner task and configured teacher/student obs dims.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds the ON contract for `task=g1_stand_still/mujoco`, proving the sentinel
    creates a 98-D temp student checkpoint and routes it through distill
    playback.

Feature-flag contract:

```text
flag name: --task
OFF behavior: default remains `g1_walk_height/mujoco`, 99-D student playback.
ON behavior: `--task g1_stand_still/mujoco` composes the 98-D stand-still
  distill owner and creates the real `G1StandStill` playback env.
forbidden mixed state caught by probe: 98-D student checkpoint with default
  99-D height task raises `Student obs dim mismatch: expected 98, got 99`.
tests proving OFF: existing default distill playback sentinel tests.
tests proving ON: `test_g1_distill_playback_live_sentinel_stand_still_policy_checkpoint_contract`.
```

Commands and evidence (2026-07-10):

```bash
uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --steps 1 --action-mode policy --load-run /private/tmp/unilab-step321-stand-student.pt --device cpu
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill_playback_live_sentinel"
uv run ruff check scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py tests/scripts/test_train_scripts.py
uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --task g1_stand_still/mujoco --steps 2 --action-mode policy --load-run /private/tmp/unilab-step321-stand-student.pt --device cpu
```

Results:

- Negative preflight with the default height task: FAIL as expected with
  `Student obs dim mismatch: expected 98, got 99`. This confirms the previous
  sentinel entrypoint was unsafe for the 98-D standing checkpoint unless the
  owner task is selected explicitly.
- Contract tests: PASS, `4 passed, 166 deselected`.
- Ruff focused suite: PASS.
- Real stand-still student live sentinel: PASS with
  `distill_playback/task=G1StandStill`,
  `distill_playback/task_owner=g1_stand_still/mujoco`,
  `cfg_student_obs_dim=98`, `cfg_teacher_obs_dim=98`,
  `policy_obs_mode=actor`, `checkpoint_path=/private/tmp/unilab-step321-stand-student.pt`,
  `action_dim=29`, `physics_shape=[1, 72]`, `actions_shape=[1, 29]`,
  `actions_abs_max=0.0806829035282135`, and
  `[PASS] distill_playback/policy_action_nonzero`.

Status: COMPLETE for 98-D stand-still trained-student MuJoCo reset/step
connectivity. Policy quality, viewer-window behavior, and formal training
lifecycle remain unclaimed.

### Step 3.23: Formal Distill Run Lifecycle

Scope: turn the bounded saved-dataset update into a repeatable formal run
lifecycle with run metadata, final checkpoint layout, and reload/playback proof.

Non-scope:

- no online replay buffer or async distillation loop;
- no policy-quality or standing-stability claim;
- no viewer-window validation;
- no MoE expert semantics.

Files:

- Modified: `conf/distill/config.yaml`
  - Adds inert defaults: `training.formal_run=false`,
    `training.formal_run_name=null`, `training.formal_run_dir=null`.
- Modified: `scripts/train_distill.py`
  - Adds `run_formal_offline_dataset_update`.
  - Uses `ExperimentTracker` to write `run_config.json` and
    `run_summary.json`.
  - Saves the final student checkpoint in the run dir as
    `model_<samples_seen>.pt`.
  - Routes Hydra main into formal lifecycle only when
    `training.formal_run=true` and `training.offline_dataset_path` is set.
- Modified: `tests/config/test_config_system.py`
  - Adds OFF/default and override coverage for the formal-run fields.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds a 98-D stand-still formal-run contract test that verifies metadata,
    checkpoint naming, and student reload.

Feature-flag contract:

```text
flag name: training.formal_run
OFF behavior: existing collect/dry-run/offline update branches are unchanged;
  `training.offline_checkpoint` still controls the old saved-dataset probe.
ON behavior: with `training.offline_dataset_path`, the entrypoint writes a
  formal run dir containing `run_config.json`, `run_summary.json`, and
  `model_<samples_seen>.pt`.
generated/derived overrides: `training.formal_run_dir` wins over
  `training.log_dir`; otherwise `logs/distill/<task>/<timestamp>_<backend>` is
  used.
forbidden mixed states: formal run without `training.offline_dataset_path`
  raises before training.
tests proving OFF: `test_distill_g1_walk_height_owner_composes`.
tests proving ON:
  `test_distill_moe_student_config_composes_only_when_selected`,
  `test_distill_script_formal_stand_still_run_writes_metadata_and_checkpoint`.
```

Commands and evidence (2026-07-10):

```bash
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/scripts/test_train_scripts.py -q -k "formal_stand_still or teacher_policy"
uv run ruff check scripts/train_distill.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill and not hora"
uv run scripts/train_distill.py task=g1_stand_still/mujoco training.collect_dataset_path=/private/tmp/unilab-step323-stand-teacher-policy-dataset.pt training.collect_num_samples=3 training.collect_num_envs=1 training.collect_action_mode=teacher_policy
uv run scripts/train_distill.py task=g1_stand_still/mujoco training.formal_run=true training.formal_run_dir=/private/tmp/unilab-step323-formal-run training.offline_dataset_path=/private/tmp/unilab-step323-stand-teacher-policy-dataset.pt training.offline_batch_size=2 training.offline_max_updates=1
uv run python -c "import json; from pathlib import Path; from unilab.algos.torch.distill import load_distillation_student_policy; run=Path('/private/tmp/unilab-step323-formal-run'); p=load_distillation_student_policy(run/'model_2.pt', device='cpu'); cfg=json.loads((run/'run_config.json').read_text()); summ=json.loads((run/'run_summary.json').read_text()); print({'student_obs_dim': p.obs_dim, 'student_action_dim': p.action_dim, 'agent_steps': p.agent_steps, 'run_task': cfg['run']['task'], 'formal_run': cfg['config']['training']['formal_run'], 'summary_status': summ['status'], 'summary_samples_seen': summ['samples_seen'], 'summary_checkpoint': summ['checkpoint_path']})"
uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --task g1_stand_still/mujoco --steps 1 --action-mode policy --load-run /private/tmp/unilab-step323-formal-run/model_2.pt --device cpu
```

Results:

- Config distill suite: PASS, `5 passed, 117 deselected`.
- Formal/teacher-policy script focused tests: PASS, `3 passed, 168 deselected`.
- Script distill focused suite: PASS, `22 passed, 149 deselected`.
- Ruff focused suite: PASS.
- First formal live attempt failed before training because the prior
  `/private/tmp/unilab-step321-stand-teacher-policy-dataset.pt` artifact had
  been removed. The dataset was regenerated with the same owner route as
  `/private/tmp/unilab-step323-stand-teacher-policy-dataset.pt`.
- Real teacher-policy dataset regeneration: PASS with `dataset_student_obs_dim=98`,
  `dataset_teacher_obs_dim=98`, `env_steps=2`,
  `action_abs_max=0.13842099905014038`, and `synthetic_teacher_tail=False`.
- Real formal run: PASS with `distill_source=formal_offline_dataset`,
  `student_obs_shape=(2, 98)`, `teacher_obs_shape=(2, 98)`,
  `teacher_action_shape=(2, 29)`, `teacher_action_requires_grad=False`,
  `update_count=1`, `samples_seen=2`, finite loss, run dir
  `/private/tmp/unilab-step323-formal-run`, and checkpoint
  `/private/tmp/unilab-step323-formal-run/model_2.pt`.
- Metadata/reload probe: PASS with `student_obs_dim=98`,
  `student_action_dim=29`, `agent_steps=2`, `run_task=G1StandStill`,
  `formal_run=True`, `summary_status=completed`, and
  `summary_samples_seen=2`.
- Formal checkpoint live playback: PASS with `G1StandStill`,
  `cfg_student_obs_dim=98`, `action_dim=29`, `physics_shape=[1, 72]`,
  `actions_shape=[1, 29]`, and `policy_action_nonzero=0.123931`.

Status: COMPLETE for repeatable 98-D stand-still formal saved-dataset
distillation lifecycle. Viewer-window behavior, policy quality, replay/online
training, and MoE expert semantics remain unclaimed.

### Step 3.24: Stand-Still Distill Viewer Path Preflight

Scope: make the human-inspection route scriptable for the formal 98-D
stand-still student checkpoint before opening a GUI window.

Non-scope:

- no claim about standing quality or long-horizon stability;
- no replay/online training;
- no MoE expert-role diagnosis;
- no automatic viewer-window interaction.

Owner boundary:

```text
/private/tmp/unilab-step323-formal-run/model_2.pt
  -> conf/distill task=g1_stand_still/mujoco
  -> scripts/play_interactive.py --algo distill arg/build path
  -> create_distill_playback_session(...)
  -> policy action from 98-D actor obs
  -> _load_viewer_model(...)
  -> MuJoCo MjData + mj_setState/mj_forward
```

Files:

- Created: `scripts/deploy/check_unilab_g1_distill_viewer_path.py`
  - Preflights the generic distill checkpoint path up to viewer model loading
    and physics-state transfer.
  - Prints the exact `mjpython scripts/play_interactive.py --algo distill ...`
    command for manual inspection.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds a fake-session contract proving the checker routes through the
    stand-still 98/98 config, checkpoint path, nonzero policy action, viewer
    model load, and state-transfer hook.

Commands and evidence (2026-07-10):

```bash
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill_viewer_path or distill_playback_live_sentinel"
uv run scripts/deploy/check_unilab_g1_distill_viewer_path.py \
  --task g1_stand_still/mujoco \
  --load-run /private/tmp/unilab-step323-formal-run \
  --checkpoint model_2.pt \
  --action-mode policy \
  --device cpu
```

Results:

- Focused script tests: PASS, `5 passed, 167 deselected`.
- Viewer-path live preflight: PASS with `task=G1StandStill`,
  `task_owner=g1_stand_still/mujoco`, `cfg_student_obs_dim=98`,
  `cfg_teacher_obs_dim=98`, checkpoint
  `/private/tmp/unilab-step323-formal-run/model_2.pt`, `policy_obs_mode=actor`,
  `physics_shape=[1, 72]`, `actions_shape=[1, 29]`,
  `policy_action_nonzero=0.113281`, viewer model `nq=36`, `nv=35`, `nu=29`,
  and `state_transfer={'qpos_shape': [36], 'qvel_shape': [35], 'ctrl_shape': [29]}`.
- `mjpython` is available at `.venv/bin/mjpython`.
- Manual viewer command printed by the preflight:

```bash
mjpython scripts/play_interactive.py --algo distill --task g1_stand_still --sim mujoco algo.load_run=/private/tmp/unilab-step323-formal-run interactive.action_mode=policy algo.checkpoint=model_2.pt training.device=cpu
```

Status: COMPLETE for scriptable viewer-path preflight. Actual human visual
inspection still requires launching the MuJoCo viewer window and closing it
manually; do not treat that as completed from this preflight alone.

### Old Step 24b: Actual GUI Human Inspection

Scope: launch the real MuJoCo viewer path for the formal 98-D stand-still
student checkpoint and verify that the process reaches the GUI boundary without
crashing.

Non-scope:

- no claim about visual standing quality or long-horizon stability;
- no automated pixel/video assertion;
- no MoE expert-role interpretation.

Owner boundary:

```text
/private/tmp/unilab-step323-formal-run/model_2.pt
  -> mjpython scripts/play_interactive.py --algo distill
  -> create_distill_playback_session(...)
  -> G1StandStill reset/step with policy action
  -> _load_viewer_model(...)
  -> mujoco.viewer.launch_passive(...)
```

Commands and evidence (2026-07-10):

```bash
uv run scripts/deploy/check_unilab_g1_distill_viewer_path.py \
  --task g1_stand_still/mujoco \
  --load-run /private/tmp/unilab-step323-formal-run \
  --checkpoint model_2.pt \
  --device cpu

uv run python -c '<launch mjpython scripts/play_interactive.py ...; hold 15s; terminate>'
```

Results:

- Viewer-path live preflight: PASS with `task=G1StandStill`,
  `task_owner=g1_stand_still/mujoco`, `cfg_student_obs_dim=98`,
  `cfg_teacher_obs_dim=98`, checkpoint
  `/private/tmp/unilab-step323-formal-run/model_2.pt`,
  `policy_obs_mode=actor`, `physics_shape=[1, 72]`,
  `actions_shape=[1, 29]`, `policy_action_nonzero=0.110905`,
  viewer model `nq=36`, `nv=35`, `nu=29`, and finite state transfer into
  viewer `qpos/qvel/ctrl`.
- Actual GUI launch hold probe refresh (2026-07-10): PASS. The command ran
under `mjpython`, the process was still alive after 20 seconds, and the probe
terminated it intentionally. This refresh did not capture the opening/control
stdout lines, so the evidence is GUI process reachability, not visual quality or
stdout-log reachability.
- Expected macOS/Gymnasium noise observed: Gymnasium Box cast overflow warnings
  and an IMK wakeup warning in one attempted launch. They did not block the
  successful hold probe.

Status: COMPLETE for actual GUI launch-path reachability of the formal
stand-still distill checkpoint. Human visual quality remains qualitative and is
not claimed by this sentinel.

### Plan Reconciliation: Old Step 17-25 vs Current Step 3.x

Purpose: keep the original migration plan readable after the route changed from
a 99-D `G1WalkHeight` teacher to the explicit 98-D `G1StandStill` owner path.
The `Step 3.x` labels are execution-log labels, not replacements for the old
research-plan boundaries.

| Old step | Original boundary | Current evidence | Status | Remaining gap |
| --- | --- | --- | --- | --- |
| Old Step 17 | Teacher obs contract audit | Covered by the teacher-source audit and the later explicit `g1_stand_still/mujoco` owner route; final accepted route is 98-D teacher/student obs with identity projection and `teacher_policy_checkpoint_path=logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt`. | Complete for the 98-D standing route. | Not a valid 99-D/100-D height teacher; do not reuse this checkpoint for `g1_walk_height`. |
| Old Step 18 | Live rollout dataset collection | Current Step 3.20/3.21 collected real `G1StandStill` MuJoCo datasets with `env_steps>0`, 98/98 dims, `synthetic_teacher_tail=False`; Step 3.21 selected teacher-policy collection as the meaningful nonzero route. | Complete for bounded stand-still live dataset collection. | No replay buffer or long rollout dataset. |
| Old Step 19 | Action source contract | Step 3.17 added `random`; Step 3.21 added guarded `teacher_policy` only for `g1_stand_still/mujoco`. | Complete for zero/random/teacher-policy contracts. | No student-policy or replay/action-replay action source. |
| Old Step 20 | Teacher target runtime path | Step 3.21 saved-dataset offline update used frozen 98-D teacher action targets with `teacher_action_requires_grad=False`, shape `(2, 29)`, and student-only gradient/update. | Complete for bounded saved-dataset update. | No online/replay recomputation loop. |
| Old Step 21 | Formal distill run lifecycle | Step 3.23 wrote `run_config.json`, `run_summary.json`, and `/private/tmp/unilab-step323-formal-run/model_2.pt`, then reloaded and live-played the formal checkpoint. | Complete for short formal saved-dataset lifecycle. | No long training-quality claim. |
| Old Step 22 | Trained student live sentinel | Step 3.22 ran `/private/tmp/unilab-step321-stand-student.pt` through real `G1StandStill` MuJoCo reset/step with finite nonzero policy action; Step 3.23 repeated the live playback boundary for `/private/tmp/unilab-step323-formal-run/model_2.pt`. | Complete. This was not skipped. | No policy-quality or standing-stability claim. |
| Old Step 23 | MoE expert semantics | Local MoE contracts exist earlier in the note. The new diagnostic helper proves toy role separation and collapse detection, and the collected stand-still dataset sentinel reports usage/entropy/collapse for a bounded MoE checkpoint. | Partially complete. | The real `G1StandStill` dataset has no role labels or command/height/velocity diversity, so real expert role separation is still unproven. |
| Old Step 24 | Viewer / human inspection path | Step 3.24 added scriptable viewer-path preflight, and Old Step 24b launched the real `mjpython scripts/play_interactive.py --algo distill` GUI path for `/private/tmp/unilab-step323-formal-run/model_2.pt`; the refreshed probe stayed alive after 20 seconds under `mjpython`. | Complete for GUI launch-path reachability. | Human visual quality, stdout opening/control-log capture, and long-horizon standing behavior are still qualitative/unclaimed. |
| Old Step 25 | Final integration gate | Completed as a focused migration final gate: stale search, note/testing/semantic-object alignment, architecture atlas JSON parse/search, focused impact suite, selected live probes, and one owner-route checker fix. | Complete for the focused migration gate. | `make test-all` was not run; replay/online training, real height teacher quality, policy quality, visual quality, and real role-rich MoE semantics remain out of scope. |

Reconciled next-step choices:

```text
Do not continue with a generic "Step 3.25" label.

Choose one named old-plan boundary:

Option A: Old Step 23: MoE Expert Semantics
  Status: diagnostic surface complete, real role-rich evidence still missing.
  Next version should use role-labelled or command-diverse data, not the
  standing-only dataset alone.

Option B: Old Step 24b: Actual GUI Human Inspection
  Status: complete for GUI launch-path reachability.
  Evidence: mjpython launched play_interactive, loaded the formal checkpoint,
  and the refreshed GUI hold probe stayed alive after 20 seconds.

Option C: Old Step 25: Final Integration Gate
  Status: complete for the focused migration gate.
  Evidence: stale search, atlas JSON parse, focused impact tests, live probes,
  and final note/test inventory alignment passed.
```

### Old Step 25: Final Integration Gate

Scope: stop feature work and close the migration diff with an evidence-backed
focused integration gate.

Non-scope:

- no long SAC/student training;
- no replay/online distillation loop;
- no policy-quality, visual-quality, or real role-rich MoE claim;
- no `make test-all` claim.

Gate findings and fixes:

- Stale/contract search found no code-level 98-D/99-D owner mixing, but it did
  expose a validation-entrypoint hazard: `check_unilab_g1_distill_teacher_obs_contract.py`
  still hard-composed default `g1_walk_height/mujoco`, so passing the accepted
  98-D `G1StandStill` checkpoint incorrectly failed as `expected=99 got=98`.
- Fixed the checker at the owner boundary by adding explicit `--task` and
  `--device` support. The default remains `g1_walk_height/mujoco`, while the
  accepted standing route is now audited with
  `--task g1_stand_still/mujoco`.
- Updated `note/testing/test_inventory.md` and
  `note/testing/semantic_objects.md` so the final testing record names the
  owner-route requirement.

Coverage matrix:

| Layer | Checked evidence | Result | Remaining gap |
| --- | --- | --- | --- |
| Architecture | CodeGraph route check for `train_distill.py`, `conf/distill`, `src/unilab/algos/torch/distill`, `create_distill_playback_session`, deploy checkers, and tests | code-confirmed route ownership | CodeGraph does not prove runtime quality |
| Stale search | Searched notes/config/scripts/src/tests for `Step 3.25`, stale viewer-window claims, bad formal `load_run` path, 98-D standing checkpoint mixed into height route, legacy 100-D bridge text, TODO/TBD/placeholder/fake/synthetic/unconfirmed terms | one real checker hazard found and fixed; remaining legacy/fake/synthetic references are either historical note sections, explicit tests, or documented gaps | older historical note sections still mention their original step-local gaps |
| Atlas/note | `note/architecture/**/*.data.json` parsed successfully; architecture atlas contains `train_distill.py`, `conf/distill`, generic behavior distillation runtime, and playback/viewer ownership entries | note-confirmed and JSON-valid | no rendered HTML visual inspection in this gate |
| Contract tests | `tests/algos/test_g1_distillation_contract.py`, `tests/scripts/test_train_scripts.py -k distill`, `tests/config/test_config_system.py -k distill`, `tests/scripts/test_visualization_entrypoints.py -k distill`, `tests/visualization/test_interactive_playback.py -k distill` | PASS | not whole-repo test coverage |
| Static checks | `ruff`, `py_compile`, `git diff --check` | PASS | none for touched Python/static syntax |
| Live probes | standing teacher obs audit, formal trained-student playback sentinel, viewer-path preflight, MoE expert diagnostic | PASS with explicit 98-D owner facts | no long-horizon behavior or policy-quality claim |

Commands and evidence (2026-07-10):

```bash
uv run ruff check scripts/train_distill.py scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py scripts/deploy/check_unilab_g1_distill_viewer_path.py scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py src/unilab/algos/torch/distill src/unilab/visualization/interactive_playback.py scripts/play_interactive.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py tests/scripts/test_visualization_entrypoints.py tests/visualization/test_interactive_playback.py
uv run python -m py_compile scripts/train_distill.py scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py scripts/deploy/check_unilab_g1_distill_viewer_path.py scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py src/unilab/algos/torch/distill/__init__.py src/unilab/algos/torch/distill/checkpoint.py src/unilab/algos/torch/distill/collector.py src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/models.py src/unilab/algos/torch/distill/moe_diagnostics.py src/unilab/algos/torch/distill/moe_student.py src/unilab/algos/torch/distill/teacher.py src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/playback.py src/unilab/visualization/interactive_playback.py scripts/play_interactive.py
uv run python -c "import json; from pathlib import Path; [json.loads(p.read_text()) for p in Path('note/architecture').glob('**/*.data.json')]; print('json-ok')"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/scripts/test_visualization_entrypoints.py -q -k distill
uv run pytest tests/visualization/test_interactive_playback.py -q -k distill
uv run scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py --task g1_stand_still/mujoco --checkpoint-path /Users/chengyuxuan/ArtiIntComVis/UniLab/logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt --device cpu
uv run scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py --task g1_stand_still/mujoco --steps 2 --action-mode policy --load-run /private/tmp/unilab-step323-formal-run --checkpoint model_2.pt --device cpu
uv run scripts/deploy/check_unilab_g1_distill_viewer_path.py --task g1_stand_still/mujoco --load-run /private/tmp/unilab-step323-formal-run --checkpoint model_2.pt --device cpu
uv run scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py --task g1_stand_still/mujoco --dataset-path /private/tmp/unilab-step323-stand-teacher-policy-dataset.pt --student-checkpoint /private/tmp/unilab-step23-stand-moe.pt --device cpu --collapse-fraction 0.90
git diff --check
```

Results:

- Static gates: PASS (`ruff`, `py_compile`, architecture JSON parse, `git diff --check`).
- Algorithm contract suite: PASS (`24 passed`).
- Script distill suite: PASS (`40 passed, 133 deselected`).
- Distill config suite: PASS (`5 passed, 117 deselected`).
- Distill visualization entrypoint suite: PASS (`2 passed, 15 deselected`).
- Distill interactive playback suite: PASS (`5 passed, 12 deselected`).
- Teacher obs live audit for standing route: PASS with
  `task_owner=g1_stand_still/mujoco`, `task=G1StandStill`, student/teacher
  `98/98`, live obs `(1, 98)`, critic `(1, 101)`, and checkpoint first weight
  `net.0.weight=98`.
- Formal checkpoint live playback: PASS with action dim `29`, finite physics
  state `(1, 72)`, actions `(1, 29)`, checkpoint
  `/private/tmp/unilab-step323-formal-run/model_2.pt`, and
  `policy_action_nonzero=0.065722`.
- Viewer-path preflight: PASS with policy action nonzero `0.124941`, viewer
  model `nq=36`, `nv=35`, `nu=29`, and finite state transfer.
- MoE expert diagnostic: PASS for overall usage/entropy/collapse on the
  standing-only dataset; WARN remains expected because the dataset has no
  `role_labels`.

Status: COMPLETE for the focused migration final integration gate. The diff is
ready for human review or a heavier whole-repo gate. Do not upgrade the
remaining research claims without new evidence.

### Old Step 23: MoE Expert Semantics Diagnostics

Scope: add a durable diagnostic surface for MoE expert usage, route entropy, and
collapse detection without claiming real policy quality.

Non-scope:

- no new long training run;
- no policy-quality or standing-stability claim;
- no claim that a standing-only dataset can prove walking/height/velocity expert
  separation;
- no change to default `student.model_type=mlp`.

Concept boundary:

```text
MoE expert semantics needs role diversity.
Toy role fixture can prove the diagnostic is interpretable.
The current real dataset is G1StandStill-only, so it can only prove
overall usage/entropy/collapse, not command/height/velocity role separation.
```

Files:

- Created: `src/unilab/algos/torch/distill/moe_diagnostics.py`
  - Adds `diagnose_moe_expert_routes` and `moe_diagnostics_to_dict`.
  - Summarizes overall and per-role expert usage, expert fractions, mean route
    entropy, dominant expert, and collapse flags.
- Modified: `src/unilab/algos/torch/distill/__init__.py`
  - Exports the MoE diagnostics API.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds toy role fixture coverage for `stand`, `walk`, and `recovery`.
  - Adds collapse guard and role-label length error coverage.
- Created: `scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py`
  - Loads a MoE student checkpoint and a saved distillation dataset.
  - Reports overall and optional role-labelled usage/entropy/collapse.
  - Warns when the dataset has no `role_labels` metadata.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds a fake role-labelled dataset/checkpoint contract for the deploy checker.

Commands and evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "moe_expert_diagnostics or moe_student_policy or moe_distillation_trainer"
uv run pytest tests/scripts/test_train_scripts.py -q -k "moe_expert_semantics_checker"
uv run python -m py_compile scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py src/unilab/algos/torch/distill/moe_diagnostics.py
uv run scripts/train_distill.py task=g1_stand_still/mujoco student.model_type=moe algo.aux_loss_coef=0.1 training.offline_dataset_path=/private/tmp/unilab-step323-stand-teacher-policy-dataset.pt training.offline_batch_size=2 training.offline_max_updates=1 training.offline_checkpoint=/private/tmp/unilab-step23-stand-moe.pt
uv run scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py --task g1_stand_still/mujoco --dataset-path /private/tmp/unilab-step323-stand-teacher-policy-dataset.pt --student-checkpoint /private/tmp/unilab-step23-stand-moe.pt --device cpu --collapse-fraction 0.90
uv run scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py --task g1_stand_still/mujoco --dataset-path /private/tmp/unilab-step323-stand-teacher-policy-dataset.pt --student-checkpoint /private/tmp/unilab-step23-stand-moe.pt --device cpu --hard-routing --collapse-fraction 0.90
```

Results:

- MoE diagnostic algorithm tests: PASS, `6 passed, 18 deselected`.
- MoE diagnostic deploy-checker test: PASS, `1 passed, 172 deselected`.
- Py compile: PASS.
- Bounded real stand-still MoE update: PASS with `student_model_type=moe`,
  `student_obs_shape=(2, 98)`, `teacher_obs_shape=(2, 98)`,
  `teacher_action_requires_grad=False`, `student_grad_norm=0.062408`,
  `behavior_loss=0.002817`, `aux_loss=0.031203`,
  `expert_usage=(0.839538, 0.380250, 0.780212)`, `route_entropy=1.039980`,
  and checkpoint `/private/tmp/unilab-step23-stand-moe.pt`.
- Soft-route real collected-dataset diagnostic: PASS/WARN with no role labels,
  `expert_fraction=[0.431372, 0.197765, 0.370863]`,
  `mean_entropy=1.044709`, `dominant_expert=0`, `max_fraction=0.431372`,
  and `collapse_detected=False`.
- Hard-route real collected-dataset diagnostic: PASS/WARN with no role labels,
  `expert_fraction=[0.666667, 0.0, 0.333333]`,
  `mean_entropy=1.044709`, `dominant_expert=0`, `max_fraction=0.666667`,
  and `collapse_detected=False`.

Status: PARTIAL COMPLETE for Old Step 23. The code can now diagnose MoE expert
usage, entropy, and collapse, and the toy fixture proves role-wise semantics are
readable. The real collected `G1StandStill` dataset has no role labels and no
command/height/velocity diversity, so real expert role separation remains
unconfirmed by design rather than by missing code.

### MoE-1: Role-labelled Distillation Dataset Contract

Scope: make behavior-role identity a first-class offline dataset contract before
any new MoE training or multi-teacher collection.

Non-scope:

- no router/loss redesign;
- no long training run;
- no claim that current standing-only datasets contain walking or height roles;
- no live collection change yet.

Concept boundary:

```text
standing / walking / height-tracking conflict
  -> each sample needs a stable role label
  -> save/load/as_batch must preserve that label
  -> MoE diagnostics and future losses can reason by role instead of only shape
```

Files:

- Modified: `src/unilab/algos/torch/distill/data.py`
  - Adds explicit `DistillationTensorDataset.role_labels`.
  - Validates role-label length and rejects empty labels.
  - Persists role labels both as a top-level payload field and mirrored metadata
    for older diagnostic consumers.
- Modified: `src/unilab/algos/torch/distill/trainer.py`
  - Adds optional `DistillationBatch.role_labels` so `as_batch` slicing does
    not erase sample semantics.
- Modified: `scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py`
  - Prefers the explicit dataset field and falls back to metadata for older
    payloads.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds old unlabeled OFF coverage and role-labelled ON roundtrip/slicing
    coverage.

Evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "distillation_dataset"
uv run python -m py_compile src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/trainer.py scripts/deploy/check_unilab_g1_distill_moe_expert_semantics.py tests/algos/test_g1_distillation_contract.py
```

Results:

- Dataset contract suite: PASS, `9 passed, 17 deselected`.
- Py compile: PASS.

Status: COMPLETE for MoE-1 data contract. This proves offline role-label
persistence and batching, not role-rich live collection or trained MoE quality.

### MoE-2: Cached-target Trainer Path

Scope: let the offline distillation trainer consume cached teacher action
targets from the dataset instead of requiring one frozen teacher module during
every update.

Non-scope:

- no multi-teacher collector yet;
- no replay buffer or online training loop;
- no router/loss redesign;
- no policy-quality or role-separation claim;
- no change to the default frozen-teacher path when datasets do not contain
  cached targets.

Concept boundary:

```text
standing / walking / height-tracking teachers may have incompatible obs layouts
  -> normalize the shared student target at the action boundary
  -> cache detached teacher action targets with each sample
  -> offline MoE trainer can learn from mixed sources without pretending their
     teacher_obs tensors share one semantic layout
```

Files:

- Modified: `src/unilab/algos/torch/distill/data.py`
  - Adds optional `DistillationTensorDataset.teacher_actions`.
  - Validates cached target rank, finite values, batch size, and optional action
    dimension.
  - Persists `teacher_actions` and `teacher_action_dim` through save/load and
    preserves them through `as_batch`.
- Modified: `src/unilab/algos/torch/distill/trainer.py`
  - Adds optional `DistillationBatch.teacher_actions`.
  - Uses cached targets without calling the teacher when present.
  - Detaches cached targets and reports `teacher_action_source` as `cached` or
    `teacher`.
- Modified: `src/unilab/algos/torch/distill/offline.py`
  - Propagates the last update's target source into
    `OfflineDistillationRunResult.last_teacher_action_source`.
- Modified: `scripts/train_distill.py`
  - Emits `teacher_action_source` in the probe payload.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds cached-target trainer, dataset roundtrip, bad-contract, and offline
    update coverage.

Evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "cached_teacher_actions or offline_distillation_run_accepts_cached or offline_distillation_run_updates_and_saves_checkpoint"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run ruff check src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py scripts/train_distill.py tests/algos/test_g1_distillation_contract.py
uv run python -m py_compile src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py scripts/train_distill.py tests/algos/test_g1_distillation_contract.py
```

Results:

- Cached-target focused suite: PASS, `5 passed, 25 deselected`.
- Full distillation contract suite: PASS, `30 passed`.
- Script distill impact suite: PASS, `40 passed, 133 deselected`.
- Ruff: PASS.
- Py compile: PASS.

Status: COMPLETE for MoE-2 cached-target trainer plumbing. This proves the
offline trainer can update from saved teacher action targets without calling a
teacher module. It does not yet prove multi-teacher collection, role-rich data,
or trained MoE policy quality.

### MoE-3: Role-conditioned MoE Loss

Scope: add an optional role-conditioned router loss for MoE students so
role-labelled datasets can supervise which expert should receive each semantic
role.

Non-scope:

- no live role-rich collection yet;
- no multi-teacher dataset assembly;
- no policy-quality or real expert-separation claim;
- no deployment-time role-label input;
- no change to default MLP or MoE behavior when the feature is OFF.

Feature flag contract:

```text
flag name: `algo.role_loss_coef`
OFF behavior: default `0.0`; `role_labels` are preserved but do not enter loss.
ON behavior: `role_loss_coef>0` requires explicit `role_expert_targets` and
  batch `role_labels`; trainer applies cross-entropy on MoE router logits.
generated/derived overrides: Hydra needs `+algo.role_expert_targets={...}` when
  adding mapping keys to the default empty dict.
forbidden mixed states:
  - positive coefficient with no mapping fails at trainer construction;
  - positive coefficient with missing labels fails at update;
  - unmapped role labels fail at update;
  - non-MoE students fail because no router logits exist.
tests proving OFF: full distill suite keeps default `role_loss_coef=0.0`.
tests proving ON: toy role fixture maps stand/walk/height to expert indices and
  proves nonzero router gradient plus runtime/probe metadata.
```

Core parameter path:

```text
DistillationTensorDataset.role_labels
  -> DistillationBatch.role_labels
  -> MoEStudentOutput.router_logits
  -> role_expert_targets[label]
  -> F.cross_entropy(router_logits, target_expert)
  -> role_loss_coef * role_loss
  -> student/router gradients
  -> BehaviorDistillationStats.role_loss / role_target_count
  -> OfflineDistillationRunResult / scripts/train_distill.py probe payload
```

Files:

- Modified: `conf/distill/config.yaml`
  - Adds inert defaults `algo.role_loss_coef=0.0` and
    `algo.role_expert_targets={}`.
- Modified: `src/unilab/algos/torch/distill/trainer.py`
  - Adds role-conditioned router cross-entropy when enabled.
  - Keeps old behavior OFF-compatible and fail-closed for half-open states.
- Modified: `src/unilab/algos/torch/distill/offline.py`
  - Carries `last_role_loss` and `last_role_target_count`.
- Modified: `scripts/train_distill.py`
  - Passes role-loss config into the trainer and records role-loss probe fields.
- Modified: tests and testing notes.

Evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "role_conditioned_router_loss"
uv run pytest tests/config/test_config_system.py -q -k "distill_moe_student_config_composes_only_when_selected or distill_config"
uv run pytest tests/scripts/test_train_scripts.py -q -k "role_conditioned_moe_trainer"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run pytest tests/config/test_config_system.py -q -k distill
uv run ruff check src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py scripts/train_distill.py tests/algos/test_g1_distillation_contract.py tests/config/test_config_system.py tests/scripts/test_train_scripts.py
uv run python -m py_compile src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py scripts/train_distill.py tests/algos/test_g1_distillation_contract.py tests/config/test_config_system.py tests/scripts/test_train_scripts.py
git diff --check
```

Results:

- Role-conditioned trainer focused suite: PASS, `2 passed, 30 deselected`.
- Distill config role-loss route: PASS, `1 passed, 121 deselected`.
- Script role-loss connector: PASS, `1 passed, 173 deselected`.
- Full distillation contract suite: PASS, `32 passed`.
- Script distill impact suite: PASS, `41 passed, 133 deselected`.
- Distill config suite: PASS, `5 passed, 117 deselected`.
- Ruff: PASS.
- Py compile: PASS.
- `git diff --check`: PASS.
- Stale search for `role_loss_coef`, `role_expert_targets`, `role_loss`, and
  `role_target_count`: all matches are in the intended config, trainer,
  offline result, script probe, and tests.

Status: COMPLETE for MoE-3 role-conditioned loss contract. This proves the
offline MoE trainer can use role labels as router supervision under an explicit
flag. It does not prove that current real datasets contain the right role
diversity or that a trained policy has meaningful expert specialization.

### MoE-4: Multi-task Collection Adapters

Scope: add a saved-source multi-task dataset adapter that merges role-specific
`DistillationTensorDataset` files into one cached-target dataset for MoE
training.

Non-scope:

- no multi-env live collection scheduler;
- no new teacher checkpoint selection or quality claim;
- no 98-D stand to 99-D height automatic projection;
- no training loop or policy-quality claim;
- no replay storage.

Feature flag contract:

```text
flag name: `training.multitask_dataset_path`
OFF behavior: default `null`; existing collect/offline/dry-run paths are unchanged.
ON behavior: save a merged dataset to `training.multitask_dataset_path` using
  `training.multitask_sources`, then exit before teacher checkpoint resolution.
source contract: each source must define `path` and `role`.
forbidden mixed states:
  - empty sources fail;
  - missing role/path fails;
  - source without cached `teacher_actions` fails;
  - source student/teacher/action dims must match current owner config.
tests proving OFF: full distill config/script/algorithm suites.
tests proving ON: toy saved datasets merge into role-labelled cached-target
  dataset and reload through dim guards.
```

Core parameter path:

```text
training.multitask_sources[{path, role}]
  -> load_distillation_dataset(... dim guards ...)
  -> require source.teacher_actions
  -> concatenate student_obs / teacher_obs / teacher_actions
  -> generate role_labels by source role
  -> metadata source_roles / source_sample_counts / source_paths
  -> save_distillation_dataset(training.multitask_dataset_path)
  -> reloadable cached-target role-labelled dataset
```

Files:

- Modified: `conf/distill/config.yaml`
  - Adds inert defaults `training.multitask_dataset_path=null` and
    `training.multitask_sources=[]`.
- Modified: `src/unilab/algos/torch/distill/data.py`
  - Adds `build_multitask_distillation_dataset`.
  - Requires cached teacher actions and matching dims before concatenation.
- Modified: `src/unilab/algos/torch/distill/__init__.py`
  - Exports the adapter.
- Modified: `scripts/train_distill.py`
  - Adds `run_multitask_dataset_assembly`.
  - Handles `training.multitask_dataset_path` before live collect/offline train
    and before teacher checkpoint resolution.
- Modified: tests and testing notes.

Evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "multitask_distillation_dataset_adapter"
uv run pytest tests/scripts/test_train_scripts.py -q -k "builds_multitask_dataset_from_saved_sources"
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run ruff check src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/__init__.py scripts/train_distill.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py
uv run python -m py_compile src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/__init__.py scripts/train_distill.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py
git diff --check
```

Results:

- Data adapter focused suite: PASS, `2 passed, 32 deselected`.
- Script adapter connector: PASS, `1 passed, 174 deselected`.
- Distill config suite: PASS, `5 passed, 117 deselected`.
- Full distillation contract suite: PASS, `34 passed`.
- Script distill impact suite: PASS, `42 passed, 133 deselected`.
- Ruff: PASS.
- Py compile: PASS.
- `git diff --check`: PASS.
- Stale search for `multitask_dataset_path`, `multitask_sources`,
  `multitask_adapter`, `build_multitask_distillation_dataset`, `source_roles`,
  and `source_sample_counts`: all matches are in config, data owner, script
  connector, and tests.

Status: COMPLETE for MoE-4 saved-source multi-task adapter. This proves
role-labelled cached-target datasets can be assembled from multiple saved
sources under one explicit entrypoint. It does not prove live multi-task
collection, teacher quality, or trained MoE specialization.

### MoE-5: Runtime Probe

Scope: add a durable runtime probe that crosses the saved-source adapter,
merged role-labelled cached-target dataset reload, and one bounded MoE offline
update.

Non-scope:

- no MuJoCo reset/step;
- no teacher training or teacher-quality claim;
- no cross-owner projection between 98-D standing and 99-D walking/height data;
- no policy-quality or final expert-specialization claim.

Runtime path:

```text
saved source datasets `{path, role}` with cached teacher_actions
  -> scripts/train_distill.py run_multitask_dataset_assembly
  -> merged DistillationTensorDataset role_labels + teacher_actions
  -> load_distillation_dataset dim guards
  -> MoEStudentPolicy
  -> BehaviorDistillationTrainer(role_loss_coef>0)
  -> run_offline_distillation_updates
  -> teacher_action_source=cached, role_loss, grad_norm
```

Files:

- Added: `scripts/deploy/check_unilab_g1_distill_multitask_runtime_probe.py`
  - Builds three toy saved-source datasets for `stand`, `walk_height`, and
    `height`.
  - Uses a raising teacher sentinel so the probe fails if cached targets are
    bypassed.
  - Prints stable facts for sample count, role counts, cached target source,
    role loss, action shapes, and student grad norm.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds direct coverage for the probe's `run_check` path.
- Modified: testing notes.

Evidence (2026-07-10):

```bash
uv run pytest tests/scripts/test_train_scripts.py -q -k multitask_runtime_probe
uv run scripts/deploy/check_unilab_g1_distill_multitask_runtime_probe.py --work-dir /private/tmp/unilab-moe5-runtime-probe
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run ruff check scripts/deploy/check_unilab_g1_distill_multitask_runtime_probe.py scripts/train_distill.py src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py tests/scripts/test_train_scripts.py tests/algos/test_g1_distillation_contract.py
uv run python -m py_compile scripts/deploy/check_unilab_g1_distill_multitask_runtime_probe.py scripts/train_distill.py src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py tests/scripts/test_train_scripts.py
git diff --check
```

Results:

- Runtime probe focused test: PASS, `1 passed, 175 deselected`.
- CLI runtime probe: PASS with `merged_num_samples=6`,
  `role_counts={height: 1, stand: 2, walk_height: 3}`,
  `teacher_action_source=cached`, `role_loss=0.750055`,
  `role_target_count=3`, and positive student grad norm.
- Full distillation contract suite: PASS, `34 passed`.
- Script distill impact suite: PASS, `43 passed, 133 deselected`.
- Ruff: PASS.
- Py compile: PASS.
- `git diff --check`: PASS.

Status: COMPLETE for MoE-5 runtime connector proof. This proves the current
MoE distillation path can run from multiple saved role-labelled cached-target
sources through one offline MoE update. It still does not prove real
multi-policy dataset quality, live multi-task collection, or final policy
behavior.

### MoE-6: Real Walking/Standing Teacher Sources

Scope: connect the real 98-D `G1WalkFlat` and `G1StandStill` SAC teacher
checkpoints into the cached-target multi-task MoE path.

Non-scope:

- no height teacher source yet;
- no automatic projection between 98-D flat/standing and 99-D height routes;
- no long training or policy-quality claim;
- no visual quality or long-horizon MuJoCo claim.

Owner flag / config contract:

```text
owner route: task=g1_walk_flat/mujoco
OFF behavior: default task=g1_walk_height/mujoco remains unchanged; height route
  still rejects teacher_policy collection because it is 99-D.
ON behavior: explicit g1_walk_flat/mujoco uses the 98-D flat-walking SAC teacher
  checkpoint and permits teacher_policy collection under the same 98-D identity
  projection guard as standing.
forbidden mixed states:
  - teacher_policy collection on G1WalkHeight;
  - teacher.task_name != training.task_name;
  - non-98-D teacher/student obs for teacher_policy collection;
  - missing cached teacher_actions for multi-task source assembly.
```

Real checkpoints:

```text
walk_flat:
  /Users/chengyuxuan/ArtiIntComVis/UniLab/logs/fast_sac/G1WalkFlat/2026-07-09_02-48-58_mujoco/model_5000.pt
  actor_input_dim=98, action_dim=29
stand:
  /Users/chengyuxuan/ArtiIntComVis/UniLab/logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt
  actor_input_dim=98, action_dim=29
```

Runtime path:

```text
task=g1_walk_flat/mujoco + walking SAC teacher
task=g1_stand_still/mujoco + standing SAC teacher
  -> teacher_policy live collection
  -> DistillationTensorDataset.teacher_actions cached at 29-D action boundary
  -> multitask source roles {walk_flat, stand}
  -> merged role-labelled 98/98 cached-target dataset
  -> MoEStudentPolicy offline update with role_loss
  -> raising teacher sentinel remains uncalled
```

Files:

- Added: `conf/distill/task/g1_walk_flat/mujoco.yaml`
  - Explicit 98-D flat-walking distill owner config.
- Modified: `src/unilab/algos/torch/distill/collector.py`
  - Persists cached `teacher_actions` when `action_mode=teacher_policy`.
- Modified: `scripts/train_distill.py`
  - Expands teacher-policy collection guard to explicit 98-D
    `G1WalkFlat`/`G1StandStill` routes.
- Added: `scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py`
  - Runs checkpoint-dim preflight, short walking/standing teacher-policy
    collection, multi-task assembly, and bounded cached-target MoE update.
- Modified: tests and testing notes.

Evidence (2026-07-10):

```bash
uv run python -c "from unilab.algos.torch.distill import inspect_sac_teacher_checkpoint; paths=['logs/fast_sac/G1WalkFlat/2026-07-09_02-48-58_mujoco/model_5000.pt','logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt']; [print(p, inspect_sac_teacher_checkpoint(p)) for p in paths]"
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "teacher_policy_action_mode or cached_teacher_actions or multitask_distillation_dataset_adapter"
uv run pytest tests/config/test_config_system.py -q -k distill
uv run pytest tests/scripts/test_train_scripts.py -q -k "walk_flat_teacher_policy or stand_still_teacher_policy or rejects_teacher_policy_collection_on_height_route or builds_walk_flat or builds_stand_still"
uv run scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py --work-dir /private/tmp/unilab-moe6-dual-teacher --num-samples 4 --num-envs 1 --device cpu
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run ruff check scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py scripts/deploy/check_unilab_g1_distill_multitask_runtime_probe.py scripts/train_distill.py src/unilab/algos/torch/distill/collector.py src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py
uv run python -m py_compile scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py scripts/deploy/check_unilab_g1_distill_multitask_runtime_probe.py scripts/train_distill.py src/unilab/algos/torch/distill/collector.py src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/config/test_config_system.py
git diff --check
```

Results:

- Checkpoint preflight: both `G1WalkFlat` and `G1StandStill` checkpoints report
  `actor_input_dim=98`.
- Focused distill data tests: PASS, `7 passed, 27 deselected`.
- Distill config suite: PASS, `6 passed, 117 deselected`.
- Focused script owner/collection tests: PASS, `5 passed, 173 deselected`.
- Real dual-teacher MoE probe: PASS with walking `action_abs_max=0.898535`,
  standing `action_abs_max=0.148596`, merged `role_counts={stand: 4,
  walk_flat: 4}`, `teacher_action_source=cached`, `role_loss=0.960151`,
  `role_target_count=4`, and positive student grad norm.
- Full distillation contract suite: PASS, `34 passed`.
- Script distill impact suite: PASS, `45 passed, 133 deselected`.
- Ruff: PASS.
- Py compile: PASS.
- `git diff --check`: PASS.

Status: COMPLETE for MoE-6 real walking/standing source connector. This proves
the provided walking and standing teachers can produce cached 29-D targets from
their own explicit 98-D owner routes and can feed one MoE offline update. It
does not prove the final student has learned a useful fused locomotion policy.

### Step SW-3: Config / Entrypoint Contract

Scope: make the standing/walking sample-selection concept an owner-config
contract instead of a manual CLI convention.

Design delta:

- Old design: `collect_command_sample_filter` existed at the collector and
  entrypoint boundary, but task owner YAML did not decide which command intent
  belongs to each teacher.
- New design: `g1_walk_flat/mujoco` owns `collect_command_sample_filter=active`
  and `g1_stand_still/mujoco` owns `collect_command_sample_filter=inactive`.
- Unchanged: base `conf/distill/config.yaml` keeps
  `collect_command_sample_filter=none`, so generic and height routes remain
  inert unless a task owner opts in.
- Forbidden mixed state: do not collect walk-flat teacher samples from
  zero-command rows; do not collect stand-still teacher samples from rows with
  any velocity/yaw command.

Owner contract:

| route | owner YAML | filter | command source | threshold |
| --- | --- | --- | --- | --- |
| default / height future route | `conf/distill/config.yaml` / `g1_walk_height/mujoco` | `none` | inert | inert |
| walking teacher route | `conf/distill/task/g1_walk_flat/mujoco.yaml` | `active` | `info["commands"]` | xy/yaw `0.05` |
| standing teacher route | `conf/distill/task/g1_stand_still/mujoco.yaml` | `inactive` | `info["commands"]` | xy/yaw `0.05` |

Evidence (2026-07-10):

```bash
uv run pytest tests/config/test_config_system.py -q -k "distill_g1_walk_height_owner_composes or distill_g1_walk_flat_owner_composes or distill_g1_stand_still_owner_composes"
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill_script_collects_stand_still_dataset_with_owner_config or distill_script_collects_owner_filtered_walk_dataset or stand_still_teacher_policy_dataset_and_updates or walk_flat_teacher_policy_cached_dataset"
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "command_sample_filter or command_active_mask or collect_distillation_dataset_from_env"
```

Results:

- Config owner suite: PASS, `3 passed, 120 deselected`.
- Entrypoint owner collection suite: PASS, `4 passed, 175 deselected`.
- Collector command-filter suite: PASS, `18 passed, 29 deselected`.

Status: COMPLETE for SW-3 config / entrypoint contract. This proves the
standing and walking distill owner routes now activate their command-intent
sample filters from config, not from ad hoc CLI overrides. It remains an
offline/fake-env contract; real MuJoCo command lifecycle is still the next live
sentinel boundary.

### Step SW-4: Dual-Teacher Probe Uses Intent Filters

Scope: make the dual-teacher MoE probe fail unless the walking and standing
source collections use the owner command-intent filters.

Design delta:

- Old probe: collected `walk_flat` and `stand` teacher-policy datasets from the
  right owner routes, but only indirectly inherited the command filter and did
  not assert it as a probe contract.
- New probe: `scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py`
  asserts `walk_flat -> active` and `stand -> inactive` at both
  `run_collect_dataset` result level and persisted dataset metadata level.
- The probe also records `command_seen_samples` and
  `command_selected_samples` in `command_filter_contracts`.

Evidence (2026-07-10):

```bash
uv run pytest tests/scripts/test_train_scripts.py -q -k "dual_teacher_probe_requires_owner_intent_filters or g1_distill_multitask_runtime_probe"
uv run ruff check scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py tests/scripts/test_train_scripts.py
uv run python -m py_compile scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py tests/scripts/test_train_scripts.py
uv run scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py --work-dir /private/tmp/unilab-sw4-dual-teacher --num-samples 2 --num-envs 1 --batch-size 2 --max-updates 1 --device cpu
```

Results:

- Fake-env probe contract: PASS, `2 passed, 178 deselected`.
- Ruff: PASS.
- Py compile: PASS.
- Real short MuJoCo dual-teacher probe: PASS.
  - `walk_flat`: `command_sample_filter=active`,
    `command_selected_samples=2`, `command_seen_samples=2`,
    `action_abs_max=0.713141`.
  - `stand`: `command_sample_filter=inactive`,
    `command_selected_samples=2`, `command_seen_samples=2`,
    `action_abs_max=0.122038`.
  - Merged role counts: `{stand: 2, walk_flat: 2}`.
  - Cached-target update: `teacher_action_source=cached`,
    `role_loss=1.184147`, `student_grad_norm=1.814933`.

Status: COMPLETE for SW-4 dual-teacher probe intent filters. This proves the
real walking/standing teacher probe now uses command-intent filters and records
the filter evidence. It still does not prove long-horizon fused policy quality.

### Step CR-1: Command Intent Contract

Scope: record the command-intent routing contract that must bind data
collection, role labels, MoE router training, checkpoint metadata, playback, and
live diagnostics.

Problem:

- Current SW-3/SW-4 proves source collection filters: walking teacher data comes
  from active velocity/yaw commands, and standing teacher data comes from
  inactive commands.
- The trained MoE playback failure showed a separate gap: the deploy-time router
  can still infer expert choice from observation state alone. In zero-command
  startup, this lets an upright walking-task state route to the walking expert
  before the robot has proven standing stability.
- Therefore the missing design object is not only `role_labels`; it is command
  intent as a first-class routing authority.

Design delta:

```text
Old design:
  role_labels identify the source teacher after collection, and
  collect_command_sample_filter selects rows for each teacher.

New design:
  command_intent is the shared semantic object across collection, dataset,
  router loss, deployment routing, and diagnostics.

Changed semantic objects:
  commands, command_active_mask, collect_command_sample_filter, role_labels,
  router_logits, route_probs, selected_expert, role_expert_targets,
  student checkpoint metadata, playback trace.

Forbidden old assumptions:
  - The MoE router may not be trusted to discover stand vs walk purely from
    proprioceptive state at deployment.
  - Zero-command rows may not train or select the walking teacher/expert.
  - Nonzero velocity/yaw rows may not train or select the standing teacher/expert.
  - A late switch to the stand expert after falling is not a valid standing
    behavior proof.

Affected phases:
  collection -> dataset persistence -> multi-task assembly -> router loss ->
  checkpoint save/load -> playback action path -> live trace.

Expected runtime evidence:
  zero command => expected_intent=stand, expected_expert=stand expert,
  selected_expert matches before falling, student-vs-standing-teacher MSE is low,
  base height remains above the standing sentinel threshold.
```

Command-intent rule:

```text
active(command)   := sqrt(vx^2 + vy^2) > xy_threshold OR abs(yaw) > yaw_threshold
inactive(command) := NOT active(command)
```

Current two-teacher routing contract:

| command intent | teacher source | role label | current target expert | allowed action source |
| --- | --- | --- | --- | --- |
| active velocity/yaw command | `G1WalkFlat` | `walk_flat` | `algo.role_expert_targets.walk_flat` (current run: `0`) | walking teacher cached action |
| inactive / no task command | `G1StandStill` | `stand` | `algo.role_expert_targets.stand` (current run: `1`) | standing teacher cached action |

Authority boundary:

- Command intent may choose or strongly bias which expert is responsible for the
  action.
- Command intent may supervise router logits during distillation.
- Command intent may be persisted as dataset metadata or per-row labels for
  diagnostics and replayable training.
- Command intent must not change reward ownership, reset physics, teacher action
  values, or hide a bad student action with a playback-only clamp.

Height-control extension:

- Height control is not part of the current two-teacher gate.
- Future height data must add an explicit third intent or sub-intent instead of
  silently overloading `walk_flat` or `stand`.
- Until a real height teacher exists, `G1WalkHeight` remains a separate 99-D
  owner route and must not be mixed into the 98-D walk/stand MoE dataset.

Required implementation steps after CR-1:

1. CR-2 Dataset command-intent schema: persist per-row command or intent labels
   and verify save/load, slicing, and multi-task assembly.
2. CR-3 Collection command owner: guarantee walking collection produces active
   commands and standing collection produces inactive commands, not only filters
   whatever the env happened to sample.
3. CR-4 Router command prior: add explicit command-intent router loss and an
   optional deploy-time hard/bias route controlled by config.
4. CR-5 Playback/live sentinel: print `expected_intent`,
   `expected_expert`, `selected_expert`, route probabilities, standing-teacher
   action MSE, and base height before the first fall.

Completion gate for the whole CR series:

- A zero-command MuJoCo playback of the fused MoE checkpoint selects or strongly
  biases to the stand expert from the first traced step.
- A nonzero velocity/yaw command selects or strongly biases to the walk expert.
- Both facts are proven by scriptable diagnostics before any visual-quality
  claim.

### Step CR-2: Dataset Schema Command/Intent

Scope: add command-intent fields to the offline distillation dataset schema so
the CR routing contract has a persistent per-row object before router or live
playback changes.

Implemented contract:

- `DistillationTensorDataset.commands`: optional `(N, 3)` tensor storing
  `[vx, vy, yaw]` command rows.
- `DistillationTensorDataset.command_intents`: optional per-row labels with
  only `active` or `inactive`.
- `DistillationBatch` preserves `commands` and `command_intents` through
  `as_batch()` and shuffled offline indexing.
- `save_distillation_dataset()` and `load_distillation_dataset()` round-trip
  the new fields while old datasets without those keys remain valid.
- `build_multitask_distillation_dataset()` concatenates command fields only when
  all sources provide them, and fails closed on mixed command-schema sources.
- `collect_distillation_dataset_from_env()` stores selected command rows and
  derived `active/inactive` intent labels when command filtering is enabled.

Non-scope:

- No MoE router prior yet.
- No deploy-time hard route yet.
- No live MuJoCo behavior claim.
- No height-control intent yet.

Evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "command_intent or command_sample_filter or collect_distillation_dataset_from_env_filters or multitask_distillation_dataset_adapter_merges"
uv run python -m py_compile src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/collector.py src/unilab/algos/torch/distill/offline.py src/unilab/algos/torch/distill/trainer.py tests/algos/test_g1_distillation_contract.py
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k "multitask_dataset or collect"
uv run ruff check src/unilab/algos/torch/distill/data.py src/unilab/algos/torch/distill/collector.py src/unilab/algos/torch/distill/offline.py src/unilab/algos/torch/distill/trainer.py tests/algos/test_g1_distillation_contract.py
```

Results:

- Focused command-intent contract: PASS, `5 passed, 47 deselected`.
- Full distillation contract suite: PASS, `52 passed`.
- Focused script connector: PASS, `9 passed, 173 deselected`.
- Py compile: PASS.
- Ruff: PASS.

Status: COMPLETE for CR-2 dataset schema. This proves command intent survives
dataset build, batch slicing, persistence, collector selection, and multitask
merge. It does not prove router behavior or live fused-policy behavior.

### Step CR-3: Collection Command Owner

Scope: fail closed when the collection entrypoint is asked to build walk/stand
distillation datasets with the wrong command-intent filter.

Implemented contract:

- `G1WalkFlat` collection requires `training.collect_command_sample_filter=active`.
- `G1StandStill` collection requires `training.collect_command_sample_filter=inactive`.
- The guard runs before env creation, so a CLI override cannot silently collect
  semantically inverted data.
- After collection, saved datasets must contain `commands`, `command_intents`,
  `command_seen_samples`, `command_selected_samples`, and exact
  `command_intent_counts`.

Non-scope:

- No router prior yet.
- No deploy-time hard or biased expert route yet.
- No claim that the trained MoE stands in MuJoCo.

Evidence (2026-07-10):

```bash
uv run pytest tests/scripts/test_train_scripts.py -q -k "owner_command_filter_override or collects_stand_still_dataset_with_owner_config or collects_owner_filtered_walk_dataset or collects_stand_still_teacher_policy_dataset_and_updates or collects_walk_flat_teacher_policy_cached_dataset"
```

Result:

- Focused script owner route: PASS, `6 passed, 178 deselected`.
- The first TDD run failed before the fix because the wrong filter reached the
  collector and failed later on fake-env dimensions, proving the missing owner
  guard was real.

Status: COMPLETE for CR-3 collection owner guard. This proves the offline
collection command route cannot be accidentally inverted by CLI overrides for
the current 98-D walk/stand MoE route. It does not prove router behavior or live
playback behavior.

### Step CR-4: MoE Router Command-Intent Supervision

Scope: add explicit command-intent router supervision to the offline MoE
distillation trainer, separate from source-role supervision.

Implemented contract:

- `algo.command_intent_loss_coef` defaults to `0.0`, so legacy distillation runs
  keep the previous behavior unless the loss is explicitly enabled.
- `algo.command_intent_expert_targets` maps `active` and `inactive` command
  intents to MoE expert indices; the current default mapping is `active: 0`,
  `inactive: 1`, matching `walk_flat -> expert 0` and `stand -> expert 1`
  while the coefficient remains off by default.
- When enabled, `DistillationBatch.command_intents` is required and is trained
  with cross-entropy on MoE `router_logits`.
- Missing target mappings, missing command-intent labels, invalid target expert
  indices, and non-MoE students fail closed.
- Offline run diagnostics and checkpoint runtime metadata record
  `command_intent_loss`, `command_intent_target_count`,
  `command_intent_loss_coef`, and `command_intent_expert_targets`.
- The dual-teacher walk/stand probe now uses `active -> walk_flat expert` and
  `inactive -> stand expert` command-intent targets in addition to role labels.

Non-scope:

- No deploy-time hard or biased expert route yet.
- No live MuJoCo playback claim.
- No height-control command intent yet.

Evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "command_intent_router_loss"
uv run pytest tests/scripts/test_train_scripts.py -q -k "dual_teacher_probe_requires_owner_intent_filters or role_conditioned_moe_trainer"
uv run pytest tests/config/test_config_system.py -q -k "distill"
uv run pytest tests/algos/test_g1_distillation_contract.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k "role_conditioned_moe_trainer or dual_teacher_probe_requires_owner_intent_filters or multitask_runtime_probe_runs_cached_moe_update or owner_command_filter_override or collects_stand_still_dataset_with_owner_config or collects_owner_filtered_walk_dataset or collects_stand_still_teacher_policy_dataset_and_updates or collects_walk_flat_teacher_policy_cached_dataset"
```

Results:

- Command-intent trainer loss: PASS, `2 passed, 52 deselected`.
- Script/probe connector: PASS, `2 passed, 182 deselected`.
- Distill config compose: PASS, `6 passed, 117 deselected`.
- Full distillation algos contract after CR-4: PASS, `54 passed`.
- Focused script impact suite after CR-4: PASS, `9 passed, 175 deselected`.

Status: COMPLETE for CR-4 offline router supervision. This proves command
intent is now a training signal for the MoE router and is recorded in offline
diagnostics. It does not prove the deployed policy selects the intended expert
in MuJoCo; that remains CR-5.

### Step CR-5: Playback/Deployment Command Routing Contract

Scope: enforce and expose the command-intent routing contract during generic
distillation playback/deployment.

Implemented contract:

- `interactive.distill_command_routing` defaults to `auto`.
- `auto` only becomes deploy-time hard routing for MoE checkpoints whose runtime
  config records `command_intent_loss_coef > 0`; old MLP checkpoints and old MoE
  checkpoints keep previous behavior.
- `hard` selects the expected expert action directly:
  `inactive -> command_intent_expert_targets.inactive` and
  `active -> command_intent_expert_targets.active`.
- `bias` keeps soft MoE mixing but adds `interactive.distill_command_routing_bias`
  to the expected expert logit.
- Playback fails closed if command routing is active but
  `env.state.info["commands"]` is missing, malformed, non-finite, or batch-size
  mismatched.
- `scripts/play_interactive.py` trace now reports routing mode, whether routing
  was applied, expected intent, expected expert, selected expert, raw selected
  expert, and routed route probabilities.
- `check_unilab_g1_distill_playback_live_sentinel.py` records and checks the
  same routing contract when a policy exposes those playback attributes.

Non-scope:

- No claim that an already-trained MoE checkpoint will stand or walk better.
- No live GUI/manual visual pass.
- No height-control expert route.

Evidence (2026-07-10):

```bash
uv run pytest tests/visualization/test_interactive_playback.py -q -k "distill"
uv run pytest tests/config/test_config_system.py -q -k "distill"
uv run pytest tests/scripts/test_train_scripts.py -q -k "distill_playback_live_sentinel"
uv run ruff check src/unilab/visualization/interactive_playback.py scripts/play_interactive.py scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py tests/visualization/test_interactive_playback.py tests/config/test_config_system.py tests/scripts/test_train_scripts.py
uv run python -m py_compile src/unilab/visualization/interactive_playback.py scripts/play_interactive.py scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py
```

Results:

- Distill playback focused suite: PASS, `8 passed, 12 deselected`.
- Distill config compose: PASS, `6 passed, 117 deselected`.
- Distill playback live sentinel focused suite: PASS, `4 passed, 180 deselected`.
- Ruff touched files: PASS.
- Py compile touched playback scripts: PASS.

Status: COMPLETE for CR-5 deployment routing contract. This proves the playback
path can force zero-command startup to the inactive/standing expert and expose
the decision in trace/sentinel output. It still does not prove trained checkpoint
quality or long-horizon MuJoCo stability.

### Step CR-6: Balanced Offline Stand/Walk Sampler

Scope: add an explicit offline batch sampler so stand/walk, or inactive/active,
training proportions are controlled per update instead of depending on saved
dataset order, dataset imbalance, or random shuffle.

Implemented contract:

- `training.offline_balance_key` defaults to `none`.
- Supported keys are `none`, `role`, and `command_intent`.
- `training.offline_balanced_labels` optionally fixes label order, for example
  `[stand, walk]` or `[inactive, active]`; if omitted, labels are inferred from
  the dataset.
- Balanced sampling is with replacement, so minority classes such as stand can
  appear in every update even when the merged dataset is imbalanced.
- Each update records `batch_label_counts`; probe output and
  `OfflineDistillationRunResult` expose the full count sequence plus the last
  batch counts.
- `distill_runtime_cfg` persists `offline_balance_key` and
  `offline_balanced_labels`, so the checkpoint records whether it was trained
  with balanced sampling.
- Formal run checkpoint naming uses `batch_size * max_updates` when balanced
  sampling is active, because samples are drawn with replacement.

Non-scope:

- No collector change.
- No live MuJoCo run.
- No claim that existing MoE checkpoints improve without retraining.
- No height-control balancing yet.

Evidence (2026-07-10):

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k "balanced_sampler or balances_role_batches or balances_command_intent"
uv run pytest tests/config/test_config_system.py -q -k "distill"
uv run pytest tests/scripts/test_train_scripts.py -q -k "offline_update_uses_balanced_role_sampler"
```

Results:

- Offline balanced sampler contracts: PASS, `3 passed, 54 deselected`.
- Distill config compose: PASS, `6 passed, 117 deselected`.
- Script/config/checkpoint connector: PASS, `1 passed, 184 deselected`.

Status: COMPLETE for CR-6 offline sampler balance. This proves balanced batches
can be constructed for both role labels and command-intent labels and that the
script route persists the setting into checkpoints. It does not prove trained
policy quality or live stability.

### Step CR-7: Dual-Teacher Runtime Probe With Balanced Updates

Scope: run the real dual-teacher walk/stand distillation probe through live
MuJoCo collection, source merge, balanced offline update, and runtime
diagnostics.

Implemented contract:

- `check_unilab_g1_distill_dual_teacher_moe_probe.py` now uses the CR-6
  balanced sampler by default with `balance_key=role` and labels
  `[walk_flat, stand]`.
- The probe payload records `balance_key`, `batch_label_counts`, and
  `last_balance_label_counts` under `offline_update`.
- The probe fails closed if balanced counts omit an expected label, do not sum
  to `batch_size`, or differ by more than 1 sample per label.
- The runtime route still verifies owner command filters:
  `walk_flat -> active`, `stand -> inactive`.
- The runtime update still uses cached 29-D teacher actions, not a live teacher
  module call.

Non-scope:

- No long training.
- No final MoE policy quality claim.
- No GUI/human inspection.
- No height-control route.

Evidence (2026-07-10):

```bash
uv run pytest tests/scripts/test_train_scripts.py -q -k "dual_teacher_probe_requires_owner_intent_filters"
uv run scripts/deploy/check_unilab_g1_distill_dual_teacher_moe_probe.py --work-dir /private/tmp/unilab-cr7-dual-teacher-probe --num-samples 4 --num-envs 1 --batch-size 4 --max-updates 2 --device cpu
```

Observed runtime facts:

- Live collection created 4 `G1WalkFlat` active samples and 4 `G1StandStill`
  inactive samples.
- Merged dataset shape was 8 samples, 98-D student obs, 98-D teacher obs, and
  cached 29-D teacher actions.
- Offline update used `teacher_action_source=cached`.
- Balanced update reported `last_counts={'walk_flat': 2, 'stand': 2}`.
- Role and command-intent router losses were both positive in the runtime probe.

Status: COMPLETE for CR-7 runtime probe. This proves the script-level
dual-teacher route now crosses live collection, persistence, balanced sampling,
and one bounded offline update. It still does not prove the trained fused policy
will stand or walk in a long MuJoCo rollout.

### Step DA-1: Distillation Dataset Audit Tool

Scope: make the large offline distillation dataset inspectable without launching
MuJoCo, retraining, or relying on one-off `/private/tmp` scripts.

Non-scope:

- no collector or trainer behavior change;
- no MoE loss or routing change;
- no checkpoint migration;
- no policy-quality claim.

Files:

- Created: `scripts/deploy/check_unilab_g1_distill_dataset_audit.py`
  - Loads a saved distillation `.pt` payload on CPU with mmap when available.
  - Reports tensor shapes/stats, role counts, command-intent counts,
    role-intent pairs, source metadata, and per-role/per-intent action stats.
  - Treats hard schema problems as `issues`; command threshold disagreement is
    reported as `warnings` so small boundary-speed samples do not masquerade as
    file corruption.
- Created: `tests/scripts/test_distill_dataset_audit.py`
  - Covers normal role/intent/source summaries, threshold mismatch warnings, and
    strict failure on hard schema row mismatch.
- Updated: `note/testing/test_inventory.md`
  - Registers the focused audit test command and the large-file smoke command.

Owner module: `src/unilab/algos/torch/distill/data.py` owns the saved dataset
payload contract; the new deploy script is a read-only diagnostic over that
contract.

Core parameter path:

```text
saved .pt payload
 -> student_obs / teacher_obs / teacher_actions / commands
 -> role_labels / command_intents
 -> metadata.source_* summaries
 -> issues / warnings / compact JSON audit
```

Test class: core parameter path for saved dataset schema and semantic labels.

Commands:

```bash
uv run pytest tests/scripts/test_distill_dataset_audit.py -q
uv run ruff check scripts/deploy/check_unilab_g1_distill_dataset_audit.py tests/scripts/test_distill_dataset_audit.py
uv run python -m py_compile scripts/deploy/check_unilab_g1_distill_dataset_audit.py tests/scripts/test_distill_dataset_audit.py
uv run python scripts/deploy/check_unilab_g1_distill_dataset_audit.py walk_stand_dagger2_merged.pt --stat-sample-rows 8192 --strict
```

Evidence (2026-07-11):

- Focused pytest: PASS (`3 passed`).
- Ruff: PASS (`All checks passed!`).
- Py compile: PASS.
- Real local `walk_stand_dagger2_merged.pt` smoke: PASS with `status=ok`,
  `num_samples=786432`, 98-D student/teacher obs, 29-D teacher actions,
  `role_counts={'stand': 393216, 'walk_flat': 393216}`,
  `command_intent_counts={'active': 393216, 'inactive': 393216}`, and no
  hard issues.
- The only warning on the real file is `3000/786432` command-intent labels
  differing from the default threshold recomputation, about 0.38 percent. This
  is a small boundary-command issue, not evidence of role/intent merge
  corruption.

Status: COMPLETE for dataset-audit tooling. Current evidence says the pulled
`walk_stand_dagger2_merged.pt` is structurally balanced and not obviously
role/intent corrupted. The next failure boundary is therefore downstream of the
dataset payload: training semantics, expert-specific imitation under hard
routing, or closed-loop deployment behavior.

### Step DA-2: Offline Student Init/Resume Path

Scope: add a real student checkpoint input path for offline distillation
continuation, so DAgger or repeated offline updates do not silently restart from
a random student.

Non-scope:

- no collector change;
- no MoE architecture or routing change;
- no replay buffer or online behavior cloning loop;
- no claim that resumed checkpoints stand or walk better in MuJoCo.

Design decision:

```text
training.offline_init_checkpoint = input student checkpoint to initialize/resume from
training.offline_resume_optimizer = whether optimizer state is restored if present
training.offline_checkpoint      = output student checkpoint to write after updates
```

Files:

- Modified: `conf/distill/config.yaml`
  - Adds inert default `training.offline_init_checkpoint: null`.
  - Adds `training.offline_resume_optimizer: true`; set it to `false` when the
    old student weights should initialize a fresh optimizer or a new learning
    rate.
- Modified: `scripts/train_distill.py`
  - Resolves `training.offline_init_checkpoint` as a real file when set.
  - Validates the checkpoint student runtime contract against current cfg before
    loading weights.
  - Loads student weights and optimizer state when present before constructing
    the offline update loop.
  - Writes `student_init_checkpoint_path`, `student_init_agent_steps`,
    `student_init_optimizer_requested`, and `student_init_optimizer_loaded` into
    probe output and saved runtime cfg.
- Modified: `src/unilab/algos/torch/distill/trainer.py`
  - Stores `student_init_metadata` on the trainer for explicit diagnostics.
- Modified: `tests/scripts/test_train_scripts.py`
  - Adds one zero-lr semantic test proving output weights come from the init
    checkpoint.
  - Adds one mismatch test proving hidden-dim/runtime cfg drift fails closed.

Owner module: `scripts/train_distill.py` owns entrypoint/trainer assembly;
`src/unilab/algos/torch/distill/checkpoint.py` remains the persistence owner.

Core parameter path:

```text
training.offline_init_checkpoint
 -> resolve file
 -> load_distillation_student_policy runtime cfg
 -> validate cfg-compatible student architecture
 -> load_distillation_checkpoint(student, optimizer)
 -> run_offline_distillation_updates(...)
 -> save output checkpoint with init provenance
```

Test class: core parameter path for checkpoint load/resume and strict
architecture guard.

Commands:

```bash
uv run pytest tests/scripts/test_train_scripts.py -q -k 'offline_update_initializes_from_student_checkpoint or offline_init_checkpoint_rejects_student_contract_mismatch'
uv run pytest tests/scripts/test_train_scripts.py -q -k 'offline_update_uses_balanced_role_sampler or saves_offline_dataset_student_checkpoint or offline_update_initializes_from_student_checkpoint or offline_init_checkpoint_rejects_student_contract_mismatch'
uv run pytest tests/scripts/test_distill_dataset_audit.py -q
uv run ruff check scripts/train_distill.py src/unilab/algos/torch/distill/trainer.py tests/scripts/test_train_scripts.py
uv run python -m py_compile scripts/train_distill.py src/unilab/algos/torch/distill/trainer.py tests/scripts/test_train_scripts.py
```

Evidence (2026-07-11):

- Focused init/resume tests: PASS (`2 passed, 187 deselected`).
- Adjacent offline update regression subset: PASS (`3 passed, 186 deselected`).
- Wider script-level distill subset: PASS (`56 passed, 133 deselected`).
- Dataset audit tests: PASS (`3 passed`).
- Ruff: PASS (`All checks passed!`).
- Py compile: PASS.

Status: COMPLETE for offline student init/resume plumbing. The next experiment
command should pass the previous student checkpoint as
`training.offline_init_checkpoint=<old_student.pt>` and a different output path
as `training.offline_checkpoint=<new_student.pt>`.

### Step DA-3: MoE Per-Expert Behavior Cloning Loss

Scope: align offline MoE behavior cloning with deployment-time command routing.
When a sample carries command intent or role labels and those labels map to an
expert, the behavior loss now supervises that expert action directly instead of
the soft mixture action.

Non-scope:

- no dataset or collector change;
- no playback hard-routing change;
- no online DAgger loop;
- no claim that the next checkpoint will automatically stand/walk without live
  validation.

Design decision:

```text
algo.expert_behavior_loss_source=auto
  command_intent labels + command_intent_expert_targets -> train selected command expert
  else role_labels + role_expert_targets                 -> train selected role expert
  else                                                   -> old student_action mixture loss
```

If role targets and command-intent targets are both present but point to
different experts for the same row, training fails closed. This protects the
core concept: zero-command samples must not silently update the walking expert,
and walking-command samples must not silently update the stand expert.

Files:

- Modified: `src/unilab/algos/torch/distill/trainer.py`
  - Adds `expert_behavior_loss_source`.
  - Extracts per-row target expert indices from command intent or role labels.
  - Uses `expert_actions[row, target_expert]` for behavior loss when available.
  - Reports `behavior_action_source`, `behavior_action_shape`, and
    `behavior_target_count`.
- Modified: `src/unilab/algos/torch/distill/offline.py`
  - Persists the last behavior-action diagnostics through offline run results.
- Modified: `scripts/train_distill.py`
  - Saves `expert_behavior_loss_source` in `distill_runtime_cfg`.
  - Prints behavior-action diagnostics in the CLI probe result.
- Modified: `conf/distill/config.yaml`
  - Adds default `algo.expert_behavior_loss_source: auto`.
- Modified: `tests/algos/test_g1_distillation_contract.py`
  - Adds a toy MoE fixture where soft mixture would be wrong but selected
    command expert imitation has zero loss.
  - Adds a fail-closed conflict test for role/command expert target mismatch.

Owner module: `src/unilab/algos/torch/distill/trainer.py` owns loss semantics.

Core parameter path:

```text
DistillationBatch.command_intents / role_labels
 -> command_intent_expert_targets / role_expert_targets
 -> target expert index per row
 -> MoEStudentOutput.expert_actions[:, target]
 -> behavior_loss
 -> OfflineDistillationRunResult / train_distill probe diagnostics
```

Test class: core parameter path for expert target selection and behavior loss.

Commands:

```bash
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k 'expert_behavior_loss or command_intent_expert_behavior_loss or conflicting_role_and_intent_targets'
uv run pytest tests/algos/test_g1_distillation_contract.py -q -k 'moe or command_intent or role_conditioned'
uv run pytest tests/scripts/test_train_scripts.py -q -k distill
uv run ruff check src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py scripts/train_distill.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py
uv run python -m py_compile src/unilab/algos/torch/distill/trainer.py src/unilab/algos/torch/distill/offline.py scripts/train_distill.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py
```

Evidence (2026-07-11):

- Focused expert-behavior tests: PASS (`2 passed, 60 deselected`).
- Wider MoE/command/role algorithm subset: PASS (`18 passed, 44 deselected`,
  with two existing zero-element tensor warnings).
- Script-level distill subset: PASS (`56 passed, 133 deselected`).
- Ruff: PASS (`All checks passed!`).
- Py compile: PASS.

Status: COMPLETE for per-expert BC loss. The next training command should keep
`algo.expert_behavior_loss_source=auto` and should show
`behavior_action_source=command_intent_expert` when using the current
walk/stand command-intent dataset.

### Step DA-4: Iterative Online DAgger Loop

Observed failure: zero-command playback routed correctly to stand expert 1,
but student-vs-standing-teacher MSE grew from `0.000417` at step 10 to
`0.038095` at step 100 as the student moved off the teacher rollout
distribution. The robot then fell. This rules out command routing as the first
failure and identifies closed-loop state-distribution drift as the owner
boundary.

Design alignment with agile-demo:

```text
agile-demo:
  student action -> env step -> current teacher action -> rollout storage -> update

UniLab DA-4:
  student action -> env step -> current teacher action -> validated tensor dataset
  -> immediate per-expert update -> updated student recollects the next iteration
```

The UniLab path keeps command-intent filters, role labels, per-expert BC, done
reset guards, checkpoint metadata, and Hydra ownership. It no longer requires a
human to manually alternate collection, merge, and retraining for every DAgger
round.

Files:

- Created: `src/unilab/algos/torch/distill/dagger.py`
  - owns iterative recollection, immediate offline updates, role attachment,
    and final checkpoint persistence.
- Modified: `scripts/train_distill.py`
  - assembles env, teacher, initialized student, and the DAgger owner loop.
- Modified: `conf/distill/config.yaml`
  - adds the OFF-by-default `training.online_dagger` route and bounded loop
    parameters.
- Modified: algorithm and script contract tests.

Core parameter path:

```text
initialized MoE student
 -> student_policy live rollout
 -> teacher labels current student states
 -> command intent / role selects expert 0 or 1
 -> immediate BehaviorDistillationTrainer updates
 -> updated student becomes the next rollout policy
 -> final checkpoint preserves prior agent_steps + new samples_seen
```

Evidence (2026-07-11):

- TDD red: owner test failed because `run_iterative_dagger_updates` did not
  exist.
- TDD green: two-iteration toy proves the second rollout uses the updated
  student (`1 passed`).
- Entrypoint contract proves Hydra parameters, role label, command filter, and
  checkpoint path reach the owner loop (`1 passed`).

Status: contract-confirmed for the online DAgger training loop. Live MuJoCo
policy quality remains a separate sentinel and is not inferred from toy tests.

### Sequential DAgger Expert Isolation Fix (2026-07-11)

The first sequential stand-then-walk DAgger path contained an optimizer-state
leak. Per-expert behavior cloning selected only the requested expert action,
but autograd still produced zero gradient tensors for the other experts. Adam
therefore continued applying their stored momentum during later single-role
updates. A walk-only DAgger stage could silently move the standing expert even
though the current standing-expert gradient was numerically zero.

The trainer now records which experts contributed behavior actions and sets
all inactive expert parameter gradients to `None` before gradient clipping and
`optimizer.step()`. This makes Adam skip inactive experts while preserving
normal updates for mixed-role batches and router parameters.

Evidence:

- TDD red reproduced the leak by first creating expert-1 Adam momentum and then
  running five expert-0-only updates; expert-1 weights changed.
- TDD green proves every expert-1 parameter remains bitwise unchanged across
  the same expert-0-only update sequence.
- Focused contract suite: `66 passed`.
- Distillation script suite: `57 passed, 133 deselected`.

Existing checkpoints produced by the leaking sequential optimizer path remain
potentially contaminated. The fix does not repair their weights in place. The
next live checkpoint must be trained from the last known-good checkpoint before
the role-specific stage, or by rerunning stand DAgger followed by walk DAgger.

### Low-Speed Routing and DAgger Aggregation Fix (2026-07-11)

`walk_stand_moe_fixed.pt` proved that standing survived expert optimizer
isolation, but its playback log exposed two separate remaining contracts:

- Playback `auto` routing incorrectly became `none` when
  `command_intent_loss_coef=0`, even though the checkpoint declared
  `expert_behavior_loss_source=command_intent`. Commands at `vx=0.2`, `0.4`,
  and `0.6` therefore stayed on the standing expert; only the learned router
  switched at `vx=0.8`.
- The iterative DAgger loop recollected student states but trained only the
  newest dataset. It did not implement the dataset aggregation that defines
  DAgger, allowing later walk-state batches to forget earlier balance states.

Playback `auto` now selects hard command-intent routing whenever either the
router loss or expert behavior source was command-intent trained. Iterative
DAgger now trains iteration `k` on the full union of datasets `1..k`, preserving
teacher actions, commands, intents, and role labels. Collection metadata records
`dagger_aggregate_iterations` and `dagger_aggregate_num_samples`.

Evidence:

- Playback regression reproduces the zero-loss-coefficient checkpoint and
  proves zero command selects stand expert 1 while `vx=0.2` selects walk expert
  0 under `auto`.
- Two-iteration DAgger regression proves training dataset sizes progress from
  4 to 8 samples instead of replacing the first four samples.
- Distillation contract suite: `66 passed`.
- Interactive playback suite: `20 passed`.
- Distillation script suite: `57 passed, 133 deselected`.

The routing fix applies immediately to existing command-intent checkpoints.
The aggregation fix changes training data exposure and therefore requires a new
walk DAgger checkpoint to evaluate long-horizon balance.

## Validation Ladder

1. Static owner scan:
   - CodeGraph for config -> env -> reward -> obs -> trainer -> storage -> loss -> playback.
2. Pure helper contract:
   - Height reward ordering and command sampling.
3. Fake-batch training contract:
   - Teacher action target, student action, loss, gradient, checkpoint keys.
4. Offline MoE routing contract:
   - Command/mode samples route to expected expert labels.
5. One-step live sentinel:
   - Env creates expected obs and reward logs.
6. One-update live sentinel:
   - Distillation trainer prints finite loss and expert usage.
7. Short visual playback:
    - Student-only checkpoint can enter real MuJoCo policy playback without
      privileged observations.
8. Viewer-path preflight:
   - The formal stand-still checkpoint can load a MuJoCo viewer model and push
     finite physics state into viewer data without opening the GUI window.
9. Human viewer inspection:
   - Actual `mjpython scripts/play_interactive.py ...` window launch and visual
     judgment remain separate from scriptable preflight.

## Risk Register

| Risk | Failure mode | Probe |
| --- | --- | --- |
| Height command changes obs dim | Old checkpoint/playback breaks silently | obs dim guard and new task name |
| Height reward dominates walking | Policy learns crouch/height hack | per-term reward lab comparison |
| Teacher and student obs mismatch | Distillation trains wrong mapping | shape guard at trainer entry |
| Teacher action detached incorrectly | Teacher receives gradients or graph retained | `requires_grad` and grad norm probe |
| MoE router collapse | One expert handles all samples | expert usage histogram and load loss |
| GPL code contamination | External GPL implementation copied into UniLab | source policy review before patch |
| MoE too early | Router learns reward bugs instead of behavior modes | Phase 1/2 stop conditions before Phase 3 |
| Teacher source drift | A `G1WalkFlat` or legacy checkpoint is treated as a real height-conditioned teacher | checkpoint metadata audit plus teacher-obs preflight before formal training |
| Command intent not enforced at deploy time | Zero-command startup routes to walking expert and falls before stand expert can recover | command-intent router trace: expected intent/expert, selected expert, standing-teacher MSE, base height |
| Training batch role drift | Stand samples are too rare inside MoE update batches, so the router/action loss is dominated by walk data | `offline_balance_key=role` or `command_intent` plus `offline_batch_label_counts` |

## Immediate Next Step

The focused migration gate is complete and the new active research boundary is
the CR command-intent series. Do not continue with generic `Step 3.x` labels;
use CR labels until command-intent routing is either implemented or explicitly
rejected.

```text
Current completed chain:
  98-D G1StandStill teacher checkpoint
  -> teacher_policy live collection
  -> saved 98/98 DistillationTensorDataset
  -> bounded offline student update
  -> reloadable 98-D student checkpoint
  -> stand-still owner live playback reset/step with finite nonzero policy action
  -> formal run dir with run_config, run_summary, model_<samples_seen>.pt
  -> formal checkpoint live playback reset/step
  -> viewer-path preflight through play_interactive distill route, viewer model,
     and MuJoCo state transfer
  -> actual mjpython GUI launch path held alive for 20 seconds in the refreshed
     probe; opening/control stdout was not captured in that refresh
  -> focused final integration gate: stale search, note/test inventory
     alignment, architecture JSON parse, impact suite, and selected live probes
Remaining named options:
  Re-train/evaluate the MoE checkpoint with command-intent loss and balanced sampling enabled
  Long-horizon MuJoCo playback and human GUI inspection
  Human review / commit preparation after CR command-intent gate
Manual GUI command verified for Old Step 24b:
  mjpython scripts/play_interactive.py --algo distill --task g1_stand_still
    --sim mujoco
    algo.load_run=/private/tmp/unilab-step323-formal-run
    algo.checkpoint=model_2.pt
    interactive.action_mode=policy
    training.device=cpu
Original inspection target:
  /private/tmp/unilab-step323-formal-run/model_2.pt under
  task=g1_stand_still/mujoco.
Stop: keep the 98-D standing route separate from the 99-D height route.
```

Do not claim trained MoE quality, formal training quality, visual standing
quality, or long-horizon viewer behavior from this migration gate.
