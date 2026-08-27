# FADA v014 Gain-Targeted Oracle Formal Runtime Audit Plan

> Status: proposed for the user-authorized formal runtime audit. Long training remains forbidden.

## Global simplified formal test

Use the unchanged `scripts/train_offpolicy.py:build_runner` composition root and resolved
`mujoco_fada_privileged_oracle` profile. The campaign has three coupled transactions under one
v014 config identity:

1. create a 32-row real MuJoCo environment, reset once, and prove that only left-knee actuator
   index `3` varies in `[0.8,1.0]`, with both nominal and attenuated rows while Actor/Critic remain
   `98/303`;
2. take one real control transition containing both zero-command and nonzero-command rows, carry
   the real Reward and observations through production `ReplayBuffer`, then execute one production
   Critic update and one Actor update;
3. use the production runner saver, strict learner reader, and Oracle gateway to restore one
   deliberately mutated full checkpoint and finalize exactly twenty intermediate checkpoints plus
   one final checkpoint.

## Critical design-point matrix

| Design point | Producer → carrier → consumer | Necessary capability | Witness | Falsifier |
|---|---|---|---|---|
| Gain-targeted source distribution | resolved Hydra profile → BackendAdapter/reset plan → MuJoCo actuator gains and typed privileged Kp/Kd scales | only left knee varies and no explicit gain enters Actor | rows contain nominal and attenuated index-3 scales; all other scales equal one; 98/303 identity | another actuator varies, range escapes, no stratum appears, or Actor dimension changes |
| Dual Reward reaches learning | G1WalkEnv transition → ReplayBuffer → privileged SAC Critic/Actor | both command modes and exact real Reward reach one update | stand/walk telemetry, reward-identical replay samples, finite metrics, Actor parameter delta | missing mode, reward mismatch, non-finite update, or unchanged Actor |
| v014 persistence and lineage | runner/learner state → checkpoint/gateway → strict reader/lineage manifest | resolved v014 identity survives save/load and exact 20+1 finalization | exact Actor restoration and 21 hashes ending at 5000 | mixed config, load mutation before validation, missing/extra checkpoint, or wrong final iteration |

## Simplifications

Use CPU, 32 environments, one reset, one control transition, zero external action, disabled
compilation, and temporary checkpoint storage. These reductions retain the official Hydra branch,
MuJoCo environment, reset owner, Reward owner, observation carrier, replay, SAC update, and
persistence owners. No test-only production hook, copied Reward, copied sampler, private-state
override, alternate learner, or persistent output is allowed.

## Evidence boundary

Local PASS can prove the v014 official local route, bounded MuJoCo reset fact, update connectivity,
and persistence. It cannot prove the server collector subprocess, CPU-pinned transfer, CUDA learner,
5000-iteration convergence, standing/walking quality, IDM/Planner quality, robustness, or
deployment. If all local edges close, exactly one server CUDA collector fact remains
`LIVE_REQUIRED` before long-training admission.

