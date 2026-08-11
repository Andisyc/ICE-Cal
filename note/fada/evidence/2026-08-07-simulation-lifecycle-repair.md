# FADA Simulation Lifecycle Repair

Date: 2026-08-07

## Scope

Checkpoint: `/Users/sss9999/locomotion/FADA/planner_idm_v005.pt`

Observed symptoms under investigation: long-running walking instability, walk-stop-walk failure, and
intermittent turning failure. No reward, Planner/IDM architecture, checkpoint, or training data was
changed.

## First Invalid Runtime Boundary

Interactive keyboard playback called `env.set_autoreset(False)`, but the stateful FADA playback
session did not receive that lifecycle change. At the environment time limit, `truncated=true` was
still returned while physics was deliberately not reset. `FADAPlaybackSession.step_once()` treated
that timeout as an actual episode reset and reset Planner-IDM histories. Because truncation remains
true on later steps, histories were repeatedly cold-started during a continuous physical trajectory.

The first bounded probe observed `truncated=true`, `terminated=false` at step 340. A second scenario
then inherited a timeout immediately. This is a playback lifecycle bug, not a policy-quality metric.

## Repair

- Playback sessions now own `set_autoreset(enabled)` and synchronize the environment mode with a
  session lifecycle flag.
- FADA history resets on `done` only when the environment will actually autoreset.
- The native keyboard/height playback entrypoint changes autoreset through the session instead of
  mutating the environment behind the session's back.
- Manual session reset still resets FADA history.

## Verification

- Contract regression proves timeout `done` does not reset FADA history when interactive autoreset is
  disabled, while enabling autoreset preserves the original episode-reset behavior.
- Focused suite: `96 passed in 1.09s`.
- Ruff: passed.
- Pyright on playback owner and entrypoint: `0 errors, 0 warnings, 0 informations`.
- Three-seed closed-loop MuJoCo probe after repair:
  - straight walking: each seed completed 1,200 commanded walking steps after a 60-step standing
    warmup; no physical termination;
  - turning: each seed completed 600 turning steps after warmup/walking; no physical termination;
  - maximum observed tilt across these runs: 14.68 degrees;
  - minimum observed base height: 0.679 m.

## Remaining Method Boundary: Walking To Standing

The repaired lifecycle exposes a separate real failure. In one bounded run, zeroing the command after
250 walking steps physically terminated at stop offset 83-84 with approximately 71-73 degrees tilt
and base height below 0.30 m.

A three-seed same-state comparison rejected the hypothesis that this is only v005 distillation error:

- the standing Oracle also fell in two of three seeds after direct takeover from the walking state;
- in one seed it failed earlier than v005;
- v005-versus-standing-Oracle stop action MSE remained small (sample maxima approximately
  `0.0026-0.0057` before the divergent failure trajectories).

Therefore the current contract's post-switch standing Oracle is not a universally executable teacher
for moving states. A deployment-only code correction cannot make this supervision valid. The next
method decision must choose either a braking/transition Oracle, a walking-Oracle deceleration phase
before standing takeover, or a newly trained transition-capable teacher. That decision changes the
training semantics and was not made in this repair.
