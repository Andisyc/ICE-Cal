# G1 Standing Routing Audit

Date: 2026-07-08

## Scope

This audit checks whether default SAC `G1WalkFlat` still enters standing-like routing when
Standing Reward is not enabled.

Non-scope:

- Do not tune reward weights.
- Do not change the explicit `+g1_walk_stage=mixed_mode` Standing Reward path.
- Do not change checkpoint or viewer compatibility logic.

## Module Checklist

| Module | Layer | Result |
|---|---:|---|
| `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml` | S0/S1 | Fixed: default Walking now explicitly sets `env.commands.small_xy_threshold: 0.0`. |
| `conf/offpolicy/g1_walk_stage/mixed_mode.yaml` | S0/S1 | Preserved: explicit Standing path still owns `mode_observation`, standing/transition ratios, `reward.mode`, and `reward.gait_constraint`. |
| `src/unilab/envs/locomotion/common/commands.py` | S1 | Preserved: global `Commands.small_xy_threshold` remains `0.2`; the fix is task-owner scoped. |
| `src/unilab/envs/locomotion/g1/joystick.py` | S1 | No code change: command sampler uses the config threshold correctly. |
| `src/unilab/training/backend_adapter.py` | S2 | No code change: override now carries only `{"small_xy_threshold": 0.0}` for default Walking. |

## Finding

Before the fix, default `G1WalkFlat` had no `env.commands` override. Therefore the runtime
fell back to `Commands.small_xy_threshold = 0.2`, so a keyboard-style `0.1 m/s` command was
zeroed into a stand-like command even though Standing Reward and mode observation were off.

Runtime probe before fix:

```text
cfg_has_env_commands= False
override_commands= None
dataclass_small_xy_threshold= 0.2
low_speed_after_default_threshold= [[0.0, 0.0, 0.0]]
```

Runtime probe after fix:

```text
cfg_commands= {'small_xy_threshold': 0.0}
override_commands= {'small_xy_threshold': 0.0}
low_speed_after_cfg_threshold= [[0.10000000149011612, 0.0, 0.0]]
mode_observation_in_cfg= False
reward_mode_in_cfg= False
```

## Fix Boundary

The fix is intentionally narrow:

- Default Walking keeps low-speed commands nonzero.
- Default Walking still does not enable `mode_observation`.
- Default Walking still does not enable `reward.mode`.
- Default Walking still does not enable `reward.gait_constraint`.
- Explicit mixed-mode Standing Reward remains behind `+g1_walk_stage=mixed_mode`.

## Validation

```text
uv run pytest tests/config/test_reward_injection.py::test_reward_config_loading_g1 tests/config/test_reward_injection.py::test_offpolicy_g1_env_override_preserves_upstream_walking_contract tests/config/test_reward_injection.py::test_g1_height_sac_config_preserves_g1_walk_flat_checkpoint_contract tests/envs/locomotion/g1/test_gait_constraint.py::test_default_g1_walking_config_keeps_keyboard_step_speed -q
4 passed
```

```text
uv run pytest tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py -q
67 passed, 2 warnings
```

```text
uv run ruff check tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py
All checks passed
```
