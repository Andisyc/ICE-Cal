# G1 Locomotion Modularization Atlas

## Problem

新训练策略仍然直接倒地, 说明当前问题不能再靠单点 reward override 猜测解决. 现在必须先把 G1 SAC locomotion 的训练, reward, observation, command, playback contract 拆成可检查模块, 再逐段加 probe.

## Current Boundary

- Default Walking: `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml`, expected clean 98-dim upstream-style Walking.
- Optional Standing: `+g1_walk_stage=mixed_mode`, expected explicit 99-dim mode-conditioned path.
- Stand-still Expert: `conf/offpolicy/task/sac/g1_stand_still/mujoco.yaml`, expected clean 98-dim zero-command standing policy path without mode routing, height command, or walking gait rewards.
- Optional height: `task=sac/g1_walk_height/mujoco` or `--height-tracking`, expected explicit 99/100-dim height-command path depending mode.
- Interactive playback: `scripts/play_interactive.py` restores checkpoint env contract before `create_sac_playback_session`.

## Atlas Files

- `note/architecture/architecture/02_g1_locomotion_modularization.data.json`
  - Repo owner map.
  - Shows current monolith blocks in `src/unilab/envs/locomotion/g1/joystick.py`.
  - Marks planned split modules: `walk_commands.py`, `walk_gait.py`, `walk_observation.py`, `walk_reward_config.py`, `walk_reward_dispatch.py`, `walk_gait_constraint.py`, `walk_gait_rewards.py`, `standing_rewards.py`, `mode_switching.py`, `walk_action.py`.
- `note/architecture/runtime/03_g1_sac_locomotion_flow.data.json`
  - Runtime flow map.
  - Separates training, default Walking, optional Standing, optional height, interactive playback, and open live failure boundary.

## Code-Confirmed Facts

- `G1WalkFlat` default config currently has no `env.mode_observation`, no `reward.mode`, no `reward.gait_constraint`, and no height command observation.
- `mixed_mode.yaml` is the explicit Standing profile and enables mode observation, command mixture, reward.mode, and gait_constraint freezing.
- `g1_stand_still/mujoco.yaml` is the explicit standalone Standing expert profile and keeps actor observation 98-dim by not enabling mode observation or height command observation.
- `g1_walk_height/mujoco.yaml` is the explicit height profile and enables `commands.observe_height_command`.
- `G1WalkEnv.obs_groups_spec` defines actor obs as `98 + mode_dim + height_dim`.
- `G1WalkEnv._compute_mode_reward` falls back to vanilla reward dispatch when `reward.mode.enabled=false`.
- `play_interactive.py` owns interactive checkpoint env contract recovery; this is separate from `train_offpolicy.py::play_offpolicy`.

## Runtime-Unconfirmed Facts

- The newly falling checkpoint has not yet been probed for reset pose, first-step action, termination reason, or per-term reward values.
- Current tests prove config/playback contracts, not physical stability.
- Direct falling may be caused by action scale/default pose/checkpoint quality/reward distribution/termination thresholds; the current atlas only isolates where each fact must be checked.

## Modularization Order

1. Extract pure command and gait helpers from `g1/joystick.py`.
   - New owners: `walk_commands.py`, `walk_gait.py`.
   - Tests: command mixture counts, gait phase target values, stand freeze behavior.
2. Extract observation contract.
   - New owner: `walk_observation.py`.
   - Tests: 98/99/100 dim matrix, symmetry layout command dim, height command column.
3. Extract reward config and dispatch.
   - New owners: `walk_reward_config.py`, `walk_reward_dispatch.py`.
   - Tests: default Walking does not execute mode reward, mixed mode masks sum to valid fractions.
4. Extract gait constraints and walking gait rewards.
   - New owners: `walk_gait_constraint.py`, `walk_gait_rewards.py`.
   - Tests: contact/height/contrast components and gate behavior.
5. Extract Standing behavior.
   - New owners: `standing_rewards.py`, `mode_switching.py`.
   - Tests: static stand, recovery stand, action authority, base-over-feet terms.
6. Extract action/phase execution.
   - New owner: `walk_action.py`.
   - Tests: phase advances in walk, freezes in stand, PD target equals default angle plus scaled action.
7. Add live failure sentinel before reward tuning.
   - Probe: reset pose, obs dim, command histogram, reward mode flags, actor action mean/std/min/max, base height, tilt, termination bit, per-term reward.

## Stop Rule

Do not adjust reward weights again until Step 7 reports the first failing boundary. The next bugfix should target the owner module named by that probe, not the whole monolithic env.
