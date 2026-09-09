# FADA v024 Command-Gated Phase-Contact Implementation Plan

Status: implementation-authorized / offline-only

Design: `FADA-METHOD-v024/FADA-TRAIN-v024/command_gated_phase_contact_v1`

## Outcome

Implement one final v024 Source Oracle profile. It includes all nine confirmed Reward decisions and
uses periodic in-episode command resampling from the first 5000-iteration training run. There is no
temporary A/B Reward implementation.

This unit does not start MuJoCo, training, deployment, checkpoint IO, or remote work. It leaves
`mujoco_fada_source` and `mujoco_fada_phase` as v022/v023 compatibility profiles.

## Engineering parameters for the first profile

- command dead zones: `0.1 m/s` for planar translation and `0.1 rad/s` for yaw;
- command-intensity saturation spans: `0.9 m/s` and `0.7 rad/s`;
- gait frequency: `0 Hz` for exact null, otherwise `0.7..1.5 Hz`;
- deterministic startup/stand phase: `[0, pi]` / `[pi, pi]`;
- duty factor and contact threshold: `0.55` and `1 N`;
- null force-balance denominator epsilon: `1e-6 N`;
- command resampling: after each completed `4 s` interval, with `30%` forced standing rows;
- Reward scales: contact mismatch `1.0`, null force imbalance `-1.0`, normalized null torque
  relaxation `-0.1`.
- domain randomization: keep targeted actuator faults outside Oracle training; retain generic
  all-joint Kp/Kd and the remaining physical DR axes.

The null force-balance cost is `abs(Fz_left-Fz_right)/max(Fz_left+Fz_right, eps)` with
both-zero force defined as cost `1`. The torque-relaxation cost is the per-row **mean** across
actuators of `(clip(tau, -tau_max, tau_max)/tau_max)^2`, not the sum. Both terms use the exact
canonical null mask and return exactly zero for every non-null row.

The torque signal is the position-servo PD torque implied by the final actuator target, post-step
joint state, per-row randomized gains, and actuator force limits. It is computed only for the v024
path, clipped to the public force range, and normalized before the null-only L2 term. Reset rows
publish an explicit zero torque with an invalid-target marker because no action has yet been
applied. Every actual v024 step must replace it from a valid final target or fail closed. This
marker is consumed after the post-step torque is materialized, so a refresh cannot replay stale
torque or masquerade as another physics step. This avoids a silent zero fallback and does not alter
v022/v023 observations or rewards.

The transition transaction is split at the physics boundary. The phase clock advances for the
command that produced the current action, and the resulting phase/contact state owns that step's
Reward. Periodic command resampling happens only after Reward calculation; the canonicalized new
command and its atomic phase transition are published in the next observation and therefore own
the next action. An external playback command follows the same canonical transition during an
observation-only refresh, without advancing the clock or resampling. A resampling boundary may
never score an old-command action against a new command.

## Owner-level steps

1. Add RED semantic tests for MTC-A through MTC-E.
2. Put final command canonicalization, exact-null detection, intensity, frequency, and G1 sampling
   in `walk_commands.py`; legacy sampling remains unchanged when v024 is disabled.
3. Put reset and four command-regime phase transitions in `walk_control.py`; the binding advances
   the current command phase before Reward, then publishes any resampled command transition only
   for the next observation/action. The legacy fixed-frequency path remains unchanged.
4. Put phase/contact mismatch, vertical-force balance, normalized torque, and clipped PD estimate
   kernels in `walk_reward.py` and `walk_control.py` according to their existing owners. Bindings
   gather only public net-contact data and already-cached numeric control state. Existing height and
   stand-contact-count terms keep their old semantics; generic Reward dispatch remains the sole
   scale/weight owner.
5. Add one `mujoco_fada_phase_contact.yaml` selector and extend the existing Oracle behavior
   identity validator. Persisted schema and tensor dimensions remain unchanged; behavior profile
   and config hashes reject v022/v023 checkpoints.
6. Run focused tests, legacy characterization tests, config/lineage tests, and architecture tests;
   repair only failures caused by this unit. Finish with an independent final maintainability review.

The bounded runtime delta audit reuses the production Hydra selector, `build_runner`,
`BackendAdapter`, and MuJoCo environment. It reduces only the environment count and rollout horizon;
it must witness exact null/non-null phase initialization, finite 98/303-D observations and Reward,
disabled left-knee-specific strength/curriculum flags, and active generic Kp/Kd randomization.

## Proof and stop boundaries

- Module tests must prove exact dead-zone boundaries, xy/yaw intensity, four phase transitions,
  per-row frequency, duty/contact truth tables, bounded negative contact Reward, force balance,
  torque normalization, in-episode resampling, old-command Reward/new-command observation
  provenance, reset-only zero torque, post-step torque refresh, and v024 identity rejection.
- Controlled counterexamples must catch translation-only frequency, synchronized non-null phase,
  radian/duty confusion, positive contact bonus, duplicated thresholds, and zero-filled torque.
- Stop on a new Reward semantic choice, a v022/v023 behavior change, private backend access, a
  checkpoint schema change, or an unrelated dirty-worktree conflict.
- Offline GREEN does not claim simulator reachability or policy quality. Training remains a later
  explicit live operation even though its intended budget is exactly 5000 iterations.
