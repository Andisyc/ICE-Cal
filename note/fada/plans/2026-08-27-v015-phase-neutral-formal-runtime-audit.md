# FADA v015 Phase-Neutral Formal Runtime Audit Plan

> Status: proposed for the user-authorized formal runtime audit. Long training remains forbidden.

## Global simplified formal test

Use the unchanged `scripts/train_offpolicy.py:build_runner` composition root and resolved
`mujoco_fada_privileged_oracle` profile. One bounded local campaign performs four coupled
transactions under one v015 identity:

1. compose the production Hydra profile and build the production privileged SAC runner;
2. create a 32-row real MuJoCo environment, reset and step once, and prove that the two legacy
   gait-phase observation slots stay exactly zero while Actor/Critic remain `98/303`;
3. prove only left-knee actuator index `3` varies in `[0.8,1.0]`, then carry real stand/walk
   Reward through production `ReplayBuffer` into one Critic and one Actor update;
4. round-trip one full checkpoint and finalize the exact 20+1 Oracle lineage through the
   production checkpoint gateway.

## Critical design-point matrix

| Design point | Producer → carrier → consumer | Necessary capability | Witness | Falsifier |
|---|---|---|---|---|
| Phase-neutral compatibility | Hydra profile → G1 reset/observation/step owners → Actor observation | legacy slots remain present but carry no gait clock | reset and post-step Actor rows have exact zeros at the two phase positions; 98-D remains | nonzero phase value, phase advance, or changed Actor dimension |
| Dual Reward reaches learning | G1 transition → ReplayBuffer → privileged SAC Critic/Actor | both command modes and exact real Reward reach one update | stand/walk telemetry, reward-identical replay samples, finite metrics, Actor delta | missing mode, Reward mismatch, non-finite update, or unchanged Actor |
| Gain-targeted source distribution | resolved profile → reset plan → MuJoCo actuator gains and privileged Kp/Kd scales | only left knee varies without entering Actor explicitly | nominal and attenuated index-3 rows; all others one; 98/303 identity | another actuator varies, range escapes, or one stratum is absent |
| Persistence and lineage | runner/learner state → checkpoint/gateway → strict reader/lineage manifest | v015 identity survives save/load and exact 20+1 finalization | exact Actor restoration and 21 hashes ending at 5000 | load mismatch, missing/extra checkpoint, or wrong final iteration |

## Simplifications

Use CPU, 32 environments, one reset, one control transition, zero external action, disabled
compilation, and temporary checkpoint storage. These reductions preserve the production Hydra
branch, MuJoCo environment, observation and Reward owners, replay, SAC updates, and persistence.
No production test hook, copied Reward, copied phase logic, private-state override, alternate
learner, or persistent output is allowed.

## Evidence boundary

PASS can prove local production-route composition, one real MuJoCo transition, one production
update, and strict persistence. It cannot prove the server collector/CUDA transfer, 5000-iteration
convergence, standing/walking quality, robustness, Planner-IDM quality, or deployment. If all local
edges close, exactly one server collector-to-CUDA fact remains `LIVE_REQUIRED` before long training.
