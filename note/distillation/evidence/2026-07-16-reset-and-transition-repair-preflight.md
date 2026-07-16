# Reset And Transition Repair Preflight

Date: 2026-07-16

Scope: fresh code audit before repairing restart-sensitive standing and
walk-to-stop balance loss. No implementation or live run was performed.

## Evidence Class

- Code-confirmed: playback lifecycle, reset command sampling, command routing,
  policy observation caching, current collector, and workflow owners.
- Runtime-confirmed from prior evidence: visible zero command previously began
  with non-zero base velocity.
- Unconfirmed: standing-teacher recovery authority on post-walk states and the
  physical contribution of one-step command/observation skew.

## Bug A: Playback Standing Reset

Current route:

```text
create playback env/session
-> playback_session.reset() samples a G1WalkFlat command
-> reset command decides gait mode and base qvel randomization
-> command-observation capability probe may execute additional resets
-> KeyboardCommander writes zero command after the final reset
```

Fresh facts:

1. `sample_g1_walk_commands()` samples from the walking distribution unless
   `commands.rel_standing_envs=1.0`.
2. `G1WalkDomainRandomizationProvider.build_reset_plan()` clears base qvel only
   for rows classified as standing from the command sampled inside that reset.
3. Writing zero before a second reset is insufficient because the reset samples
   and overwrites commands again.
4. `_policy_obs_contains_command()` can reset the env for capability detection,
   restore command/actor-observation arrays, but cannot restore the physics state
   produced by the last reset. It can therefore recreate command/physics drift.
5. `_build_keyboard_commander()` writes zero into `env.state.info["commands"]`
   after reset but does not refresh the env observation or the playback
   session's cached observation.

Correct owner boundary: interactive playback must configure its reset command
distribution as standing before the first/final reset, and external command
updates must atomically synchronize env command state, env observation, and
the playback session observation. The G1 training reset distribution remains
unchanged.

## Bug B: Walk-To-Stop Recovery

Current route:

```text
policy observation contains command_t
-> keyboard changes env.info command to zero
-> hard route reads the new env.info command
-> expert forward may still consume cached observation(command_t)
-> next env step refreshes observation
```

Fresh fact: hard routing reads `env.state.info["commands"]`, while the expert
forward consumes the playback session's cached observation. A command key event
can therefore select stand expert 1 using the new command while feeding that
expert an observation containing the previous active command. This one-step
skew is an integration defect that must be removed before measuring the
remaining training-distribution gap.

After synchronization is fixed, the structural gap remains: current DAgger
collects static standing and active walking as separate role rollouts. It does
not execute `walk -> zero command -> stand recovery`, and the dataset has no
row-level scenario or transition-age identity.

## Decision Gate

Before adding transition DAgger, compare four paired cohorts. For each source
controller and seed, replay the identical deterministic pre-switch trajectory,
then change only the post-switch controller:

```text
WT: walking teacher      -> zero -> standing teacher
WS: walking teacher      -> zero -> student stand expert
ST: student walk expert  -> zero -> standing teacher
SS: student walk expert  -> zero -> student stand expert
```

`WT vs WS` and `ST vs SS` isolate the post-switch controller. `WT vs ST`
isolates whether the state source exceeds the standing teacher's envelope. If
WT/ST cannot recover, standing-teacher labels are not a valid recovery oracle;
stop and train/acquire a recovery-capable teacher or define a curriculum. If
WT/ST recover while WS/SS fail, transition-conditioned DAgger is justified.
Blending or hysteresis is considered only if synchronized immediate standing
teacher switching still fails.

## Files Audited

- `scripts/play_interactive.py`
- `src/unilab/visualization/interactive_playback.py`
- `src/unilab/envs/locomotion/common/dr_provider.py`
- `src/unilab/envs/locomotion/g1/joystick.py`
- `src/unilab/algos/torch/distill/collector.py`
- `src/unilab/algos/torch/distill/dagger.py`
- `src/unilab/algos/torch/distill/workflow.py`
- playback, G1 gait/reset, distillation contract, and workflow tests

## Stop Condition

Do not train another candidate until Bug A and command/observation synchronization
pass deterministic and live reset gates, then the teacher recovery differential
selects or rejects transition DAgger.
