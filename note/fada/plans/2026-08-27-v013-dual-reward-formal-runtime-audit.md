# FADA v013 Dual-Reward Formal Runtime Audit Plan

> Status: proposed for the user-authorized formal runtime audit. Long training remains forbidden.

## Global simplified formal test

Use the unchanged `scripts/train_offpolicy.py:build_runner` composition root and the resolved
`mujoco_fada_privileged_oracle` profile. Build one production privileged SAC learner, create one
small real MuJoCo vector environment from the same effective config, and execute one control step.
The batch must contain both zero-command and nonzero-command rows under the configured
`rel_standing_envs=0.3` distribution.

Forward the real pre-step Actor/Critic observations, zero test action, real post-step observations,
real v013 Reward, and termination flags through the production `ReplayBuffer`, then execute one
production Critic update and one Actor update. Reuse the existing full-checkpoint strict reload and
20+1 lineage finalization transaction.

## Witnesses and falsifiers

- **Config/identity:** 98 Actor, 303 Critic, 29 Action; mode observation false; stand probability
  0.3; transition probability 0; gait scales zero; gait constraint disabled.
- **Reward branch activation:** the real batch contains at least one zero-command and one nonzero-
  command row, with structured mode telemetry reporting both branches.
- **Cross-owner effect:** replay Reward equals the environment Reward byte-for-byte and both
  production updates are finite; Actor parameters change once.
- **Persistence:** strict full-state reload restores a deliberate mutation and the gateway finalizes
  exactly twenty intermediate plus one final checkpoint.

Any identity mismatch, absent branch, non-finite Reward/update, Reward mismatch, missing update,
failed restoration, or bad lineage is a formal failure.

## Simplifications

Use CPU, a small vector environment, one real step, zero externally supplied action, disabled
compilation, and temporary checkpoint storage. These reduce cost but retain the production config,
MuJoCo environment, Reward owner, observation carrier, replay, SAC update, and persistence owners.
No production hook, private-state mutation, copied Reward, or alternate learner is allowed.

## Evidence boundary

PASS proves official local integration and persistence only. Server CUDA collector transfer remains
one possible `LIVE_REQUIRED` fact. No result proves convergence, standing/walking quality, long-run
stability, Unit B readiness, or deployment.
