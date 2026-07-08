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
- Modify: `src/unilab/algos/torch/distill/trainer.py`
  - Responsibility: add `aux_loss` and router diagnostics to behavior loss.
- Modify: `conf/distill/task/g1_walk_height/*`
  - Responsibility: expose `num_experts`, `routing_mode`, `aux_loss_coef`, and expert role labels.
- Create: `tests/algos/test_g1_moe_distillation_contract.py`
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
   - Student-only checkpoint can stand/walk without privileged observations.

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

## Immediate Next Step

Implement Phase 2 Step 2.1 only:

```text
Scope: locate the existing UniLab off-policy policy/checkpoint loading boundary and define the smallest teacher-student distillation shape contract.
Command: codegraph/query plus the smallest local contract test command selected in Step 2.1.
Expected before implementation: FAIL or missing contract because no distillation route is defined yet.
Expected after implementation: PASS with teacher obs dim, student obs dim, and teacher action dim guarded before any training loop.
Stop: distillation entry shape contract is explicit and old SAC training path is unchanged.
```

Do not start MoE until Phase 2 distillation is contract-confirmed.
Do not start distillation until the Phase 1 Completion Gate is satisfied.
