# Command-scaled phase Oracle experiment

User-approved local implementation on 2026-09-06. This is an experimental
Stage-A reward adaptation, not a claim about FADA's unpublished reward.

## Method mapping: ADAPTATION-REQUIRED

Preserve the complete existing privileged SAC pipeline: environment observations
and transitions -> replay -> privileged Actor/Critic SAC updates -> sealed 20+1
Oracle checkpoints. Preserve normalization, grouped DR, network architecture,
and the legacy phase-neutral and walking-only profiles.

Adapt only the phase reference and its objective: commanded linear velocity and
turn rate determine swing amplitude; phase determines left/right timing. The
same normalized squared height cost covers zero and nonzero commands. No stand
reward dispatcher and no actual-speed reward gate. This adapts the existing
UniLab-derived phase tracking owner; it does not reproduce a verified FADA reward.

## Engineering plan

1. Add semantic tests for stopped, moving, wrong-foot, hovering and sliding cases.
2. Add pure command amplitude, swing target and squared-error helpers in walk_math.
3. Own amplitude smoothing in environment info, updating once per action and
   initializing on reset. Reward evaluation must not mutate this state.
4. Select the new formula through reward configuration in the existing binding.
   Only the canonical flat scene is admitted. Use foot-site height corrected by
   the existing sole/site offset; retain foot-orientation cost. This reference
   point is not an exact minimum distance of a tilted sole from terrain.
5. Add a new Oracle behavior profile/config with 30% zero commands and 4-second
   command resampling. Preserve old profile identity and behavior. Do not silently
   switch Stage B/C/D to the experimental profile.
6. Run semantic, lifecycle, config and legacy regressions, then independent review.

## Confirmed acceptance oracle

At zero command, grounded feet score above gratuitously raised feet. During
swing, correct-foot target tracking scores above dragging or wrong-foot lift.
During double support, grounded feet are correct. Sliding at identical heights
is explicitly indistinguishable for this term; speed tracking retains that role.
Actual velocity cannot enable phase reward. Reset rows must not inherit amplitude
from previous episodes. Old profiles must reject the experimental reward mode.

## Limits

No new trainer, backend-private access, action limiting, training, simulation,
server operations, commit or push. Stable gait and robustness remain unverified.
Initial parameters are experimental: command scale 0.3 m/s, turn length 0.3 m,
amplitude settling time 0.15 s, height scale 0.09 m, phase cost weight 1.
