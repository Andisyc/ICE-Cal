# Command phase experiment: local implementation evidence

Scope: Stage A only. No training, playback evaluation, or policy-quality acceptance.
Existing v022/v023 task selections and downstream Stage B/C/D defaults are preserved.

## Implemented owners

- walk_math: command amplitude, alternating height reference, positive squared cost,
  first-order amplitude update. No dependence on realized velocity.
- walk_control_bindings: one amplitude update per action, stored in env info.
- walk_domain_randomization: initializes amplitude from reset commands.
- walk_reward_bindings: reads state without mutation; returns negative height cost.
- walk_config and joystick: validated new mode, canonical MuJoCo flat scene only;
  rejects terrain and fragment composition. Sole reference is the foot site plus
  local Z 2 mm rotated by foot upvector, not the tilted sole's minimum clearance.
- privileged_oracle: distinct command_phase_mixed_v1 profile, preserving sealed
  lineage checks and rejecting a new reward labeled as the old profile.
- New command_phase task YAML: 30% zero commands, 4-second command resampling,
  common stand/walk reward; no new runner, model, or standalone stand reward.

## Evidence

RED: new math tests initially failed import because implementation was absent.
GREEN: 244 tests passed across tests/envs/locomotion/g1,
tests/algos/test_fada_phase_locomotion_v023.py, and
tests/algos/test_fada_privileged_oracle_v012.py.
Existing Gymnasium float-cast overflow warnings remain (12).

New semantic tests cover grounded versus lifting/hovering at zero command;
correct alternating lift versus dragging/wrong foot; turn/reverse command;
periodicity; stopping decay; reset initialization; nonmutating negative reward;
one action update; config admission and legacy identity rejection; terrain guard.

Independent review by review_ground found the scene-composition loophole;
the terrain/fragment rejection and regression test close it.
Selected production-file Pyright: zero errors, warnings, informations.

Remaining research uncertainty: whether the learned policy stands and walks well,
reward weight tradeoffs, and stopping transient quality require actual training.
The experiment is an adaptation, not a claim of FADA's unpublished exact reward.
