---
date: 2026-08-12
evidence_class: note-confirmed
contracts: FADA-CONTEXT-METHOD-v003, FADA-CONTEXT-TRAIN-v002
runtime_status: unconfirmed
---

# Context Differentiable-Trajectory Decision

The human accepted the following method boundary:

1. Train Tracker Encoder `E` and Decoder `D` on the healthy simulated robot, then freeze both.
2. Set simulated left-knee motor strength to `0.9`, issue the same command, and collect the faulty
   probe trajectory as deployable Context input.
3. Context Encoder emits `delta_z`; the frozen Decoder executes `D(E(x) + delta_z)` in a paired
   second rollout.
4. Compare the adapted trajectory with the healthy reference trajectory.
5. Because ordinary MuJoCo does not provide the required PyTorch autograd path, train a
   differentiable fault-dynamics ensemble from MuJoCo transitions and backpropagate trajectory loss
   through that ensemble and the frozen `E/D` computation into Context only.
6. Alternate model-based Context updates with real fault-MuJoCo validation and visited-state data
   aggregation to detect and reduce model exploitation.

The decision explicitly rejects direct action labels and searched/optimized `delta_z` as semantic
Context ground truth. The first faulty trajectory is input; the healthy trajectory is reference; the
second adapted trajectory defines the loss.

This record proves only the human-selected method. No dataset, dynamics model, Context model,
gradient test, MuJoCo improvement, identifiability result, or deployment result exists yet.
