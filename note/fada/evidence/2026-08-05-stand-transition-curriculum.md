# FADA Standing And Walk-To-Stand Curriculum Evidence

Date: 2026-08-05

Scope: implementation, bounded local verification, and formal v004 source-curriculum launch.
No Oracle training was launched; existing compatible walking and standing SAC Oracles are reused.

## Implemented Evidence

- E34: `windows_per_iteration` is divided by stable largest remainder into exact `walk`,
  `static_stand`, and `walk_to_stand` quotas; default ON proportions are 50/25/25.
- E35: static standing uses a dedicated `G1StandStill` environment and standing Oracle. Walking and
  active-to-zero transition collection remain in `G1WalkFlat`.
- E36: walk-to-stand forces an active 3-D command for at least `H` steps, switches atomically to
  zero, and accepts only windows with active history and a complete zero-command future.
- E37: the persistent UniLab worker keeps the two environments, walking/standing final Oracles, and
  rollout-side Planner-IDM resident. Intermediate Oracles remain walking-only and transient.
- E38: source artifacts persist exact scenario allocations, scenario identity, Oracle role, and
  rejection counts. The parent validates these fields before replay consumption.
- E39: the ON path rejects a missing standing checkpoint, missing standing owner environment, or
  walking-Oracle fallback. The OFF path creates neither standing environment nor standing Oracle.

## Verification

- E40: focused async/curriculum suite: `23 passed`.
- E41: expanded FADA model, playback, async, and visualization suite: `51 passed`.
- E42: Ruff passed for the changed runtime/tests; focused Pyright returned zero errors.
- E43: Architecture Atlas and JSON contracts passed after adding explicit environment ownership.
- E44: real Hydra owner composition resolved `g1_stand_still/mujoco` to `G1StandStill`, MuJoCo,
  `reset_base_qvel_limit=0.0`, and `rel_standing_envs=1.0`.
- E45: local real-MuJoCo sentinel completed one async iteration with 12 windows allocated `6/3/3`,
  standing shadow-valid fraction `1.0`, no done rejection, and one expected command-crossing rejection.
- E46: strict local loading accepted both 98-D/29-D SAC Oracles. Walking SHA-256 is
  `db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`; standing SHA-256 is
  `91e18d3d1f469b2bead350cd41b33494c39c8ec8d26f2daf802e0273afa2c6da`.
- E47: formal persistent-async v004 training started at
  `/ssd1/cyx/FADA_runs/20260805_planner_idm_standing_v4`, PID `12453`, with the existing final
  walking Oracle, final standing Oracle, and 20 same-lineage intermediate walking Oracles.
- E48: iteration 1 checkpoint persisted schema 2, `196608` samples, shadow-valid fraction `1.0`,
  trajectory-IDM MSE `0.0032609`, shadow-IDM MSE `0.0032609`, and Planner-IDM Oracle-action MSE
  `0.0062462`. Seven iterations remain running asynchronously.

## Status

The v004 implementation and real connectivity gate are complete. Formal training is active on the
remote UniLab host and has completed 1/8 iterations; stability acceptance remains pending until all
iterations finish and the resulting checkpoint is evaluated in closed loop.
