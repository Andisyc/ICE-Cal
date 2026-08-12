---
status: accepted-method-design
updated_date: 2026-08-12
authority: FADA-CONTEXT-METHOD-v003
training_authority: FADA-CONTEXT-TRAIN-v002
scope: differentiable fault-model trajectory training of Context with frozen Tracker Encoder and Decoder
---

# In-Context Execution Calibration Plan

This plan records the accepted design only. No Context, differentiable dynamics, or two-rollout
runtime has been implemented or validated.

## Accepted method

```text
healthy simulation -> train and freeze Tracker Encoder E and Decoder D
healthy robot + command c -> reference trajectory tau_ref
left-knee strength 0.9 + same c + zero repair -> probe trajectory tau_probe
deployable history H(tau_probe) -> Context Encoder C -> delta_z
paired second rollout: a_t = D(E(x_t) + delta_z)
differentiable fault dynamics -> predicted adapted trajectory tau_hat_adapt
trajectory distance(tau_hat_adapt, tau_ref) -> gradient -> update only C
current C -> real MuJoCo fault validation -> aggregate transitions -> refresh dynamics model
```

The first fault trajectory is Context input, not a target. The healthy trajectory is the reference.
The second adapted trajectory defines the primary loss. Actions and free/optimized `delta_z` are not
Context labels.

## Work sequence

1. Bind the exact healthy `E/D` checkpoint, latent shape, command, left-knee `0.9` intervention, and
   reference/probe alignment lifecycle.
2. Define the fault transition schema and collect Decoder-reachable MuJoCo data from nominal,
   perturbed-latent, perturbed-action, and later current-Context rollouts.
3. Implement an ensemble fault-dynamics model with one-step and scheduled short-horizon losses.
4. Establish held-out prediction and ensemble-disagreement gates before Context optimization.
5. Implement the probe-history encoder, one-shot bounded `delta_z`, paired model rollout, and
   trajectory loss while strict-freezing `E/D/F_phi` parameters.
6. Prove gradients pass through `F_phi`, `D`, and `E` to `C`, with only Context parameters changing.
7. Alternate bounded Context model updates with paired real-MuJoCo validation and visited-state data
   aggregation to control model exploitation.
8. Evaluate zero repair, Context, constant repair, history mask, and history shuffle across
   distinguishable fault/command/phase conditions.

## Current stop condition

Do not implement or train yet. The next engineering boundary is to specify the exact checkpoint,
state/history/reference schema, paired rollout lifecycle, and numerical acceptance thresholds. The
method is `note-confirmed`; feasibility and runtime behavior remain unconfirmed.

## Open decisions

1. Exact `E/D` checkpoint and latent dimension.
2. Context history fields and probe length.
3. Reference trajectory representation and temporal alignment.
4. Dynamics model state, architecture, ensemble size, and rollout horizon.
5. Tracking, safety, latent, smoothness, and uncertainty loss weights.
6. Dataset perturbation coverage and model/real rollout update ratio.
7. Conditions needed to rule out a constant repair.
