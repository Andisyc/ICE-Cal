# G1 Walking Reward Archive

Date: 2026-07-08

## Standing-adapted walking reward archived

This is the walking reward shape that was active before restoring the original
UniLab G1 locomotion walking reward.

Source: `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml`

```yaml
reward:
  scales:
    tracking_lin_vel: 2.0
    tracking_ang_vel: 1.5
    under_speed: 0.0
    penalty_ang_vel_xy: -1.0
    penalty_orientation: -10.0
    penalty_action_rate: -4.0
    upright: 4.0
    base_height: -80.0
    pose: -0.5
    penalty_feet_ori: -20.0
    feet_phase: 5.0
    feet_phase_contrast: 0.8
    feet_phase_contact: 0.5
    alive: 10.0
  tracking_sigma: 0.12
  feet_phase_tracking_sigma: 0.004
  mode:
    balance_common_terms:
      - penalty_orientation
      - upright
      - penalty_ang_vel_xy
      - penalty_action_rate
      - base_height
      - pose
      - penalty_feet_ori
      - alive
    walk_terms:
      - tracking_lin_vel
      - tracking_ang_vel
      - feet_phase
    walk_scale_overrides:
      tracking_ang_vel: 0.3
      pose: 0.0
      penalty_action_rate: -0.5
      upright: 0.25
      penalty_orientation: -0.5
      penalty_ang_vel_xy: -0.05
      base_height: -5.0
      penalty_feet_ori: -1.0
      feet_phase: 12.0
```

## Restored walking reward target

Source: `logs/fast_sac/G1WalkFlat/2026-06-12_15-46-01_mujoco/run_config.json`

```yaml
reward:
  scales:
    tracking_lin_vel: 2.0
    tracking_ang_vel: 1.5
    penalty_ang_vel_xy: -1.0
    penalty_orientation: -10.0
    penalty_action_rate: -4.0
    pose: -0.5
    penalty_feet_ori: -20.0
    feet_phase: 5.0
    alive: 10.0
  tracking_sigma: 0.25
  feet_phase_tracking_sigma: 0.04
  mode:
    balance_common_terms: []
    walk_terms:
      - tracking_lin_vel
      - tracking_ang_vel
      - penalty_ang_vel_xy
      - penalty_orientation
      - penalty_action_rate
      - pose
      - penalty_feet_ori
      - feet_phase
      - alive
    walk_scale_overrides: {}
```
