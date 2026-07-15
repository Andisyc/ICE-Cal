# Playback Reset And Stop-Transition Root Causes

Date: 2026-07-15

Scope: the three local stand/walk MoE student candidates and interactive
MuJoCo playback under `g1_walk_flat`.

## Evidence

- E1, human live observation: all three candidates may require repeated
  launches before initial standing succeeds and frequently lose balance after
  a walking command returns to zero.
- E2, reset owner: `LocomotionDRProvider.build_reset_plan()` samples commands
  after assigning random base `qvel[:, 0:6]` in `[-limit, limit]`.
- E3, G1 reset specialization: `G1WalkDomainRandomizationProvider` clears base
  velocity only when the command sampled during reset is already classified as
  standing.
- E4, playback order: `playback_session.reset()` runs before the keyboard
  commander is created and writes its initial zero command.
- E5, runtime log: with the visible command already zero, `log.txt` recorded
  initial `linvel max_abs=0.423715` and `gyro max_abs=0.403955`.
- E6, checkpoint audit:
  - `walk_stand_moe_stand_fixed.pt` and `walk_stand_moe_aggregated.pt` have an
    identical stand expert 1 hash, `6ebce3894b92744a`;
  - `walk_stand_moe_expert_rollout.pt` has a different stand expert 1 hash,
    `a2ad119bee668c63`, but the same class of failure remains;
  - all three use command-intent hard deployment mapping `active -> 0` and
    `inactive -> 1`.
- E7, training contract: standing collection uses zero commands and zero
  standing reset base velocity; current role-specific DAgger rollouts do not
  execute a walk-to-zero transition before collecting stand recovery states.

## Issue A: Restart-Sensitive Initial Standing

Classification: integration defect, root cause confirmed.

Parameter path:

```text
G1WalkFlat reset command sample
  -> gait classification during reset
  -> random or cleared base qvel
  -> playback reset completes
  -> keyboard commander overwrites visible command with zero
  -> stand expert receives a possibly moving physical state
```

Root cause: command ownership changes after reset, but the reset-dependent
physical state is not rebuilt. The visible zero command therefore does not
prove a standing reset. Different random base velocities explain launch-to-
launch success variation independently of checkpoint identity.

Fix boundary: interactive playback initialization must establish the intended
initial command before the final reset, or perform one final reset after the
commander writes zero. The acceptance probe must record reset command and base
qvel before the first policy action.

## Issue B: Walk-To-Stop Balance Loss

Classification: training-distribution and integration gap, structural root
cause confirmed; exact recovery authority remains to be measured live.

Parameter path:

```text
walk expert 0 executes moving state
  -> command becomes zero
  -> hard routing immediately selects stand expert 1
  -> stand expert receives momentum, tilt, contact, and gait states
     absent from its standing-reset training distribution
  -> recovery action is unreliable
```

Root cause: current DAgger is role-conditioned but not transition-conditioned.
It closes each expert's local rollout distribution while leaving the
cross-role `walk -> stop -> recover standing` distribution unlabelled. Correct
expert selection therefore does not imply a correct recovery action.

Fix boundary: a future proposal must define transition-conditioned DAgger that
rolls out walking, changes the command to zero, queries the standing teacher on
the resulting student states, and updates the stand expert on the cumulative
dataset. Before activation, a live differential must establish whether the
standing teacher itself can recover those post-walk states and whether an
instant hard switch needs temporal blending or hysteresis.

## Decisions

- These are two different owners: playback reset ordering and transition-state
  training coverage.
- More static standing samples, more walking samples, or a larger checkpoint do
  not directly close either root cause.
- No candidate checkpoint is promoted by this diagnosis.
- No repair is active until its implementation and live acceptance boundary are
  approved.

## Open Risks

- The standing teacher's recovery envelope on post-walk states is unmeasured.
- The contribution of one-step observation/command skew at hard switching is
  unmeasured.
- Repeated-reset and walk-to-stop acceptance metrics are still proposals.

