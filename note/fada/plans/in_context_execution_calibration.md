---
status: accepted-method-design
updated_date: 2026-08-12
authority: FADA-CONTEXT-METHOD-v003
training_authority: FADA-CONTEXT-TRAIN-v002
scope: differentiable fault-model trajectory training of Context with frozen Tracker Encoder and Decoder
---

# In-Context Execution Calibration Plan

The differentiable core, paired MuJoCo collector, persisted dataset contract, and no-update training
preflight are implemented. Formal dynamics or Context optimization has not started.

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

The repository is at the training boundary. `planner_idm_v005.pt`, a fixed straight command, and
fixed left-knee `0.9` are bound; the collector restores one healthy MuJoCo snapshot into a same-model
fault environment so the initial physics, observation, command, and task carrier match while the
fault model retains its own gain. The preflight persists and reloads data, builds disjoint Context and
dynamics optimizers, and runs backward probes with zero optimizer steps. Do not begin formal dynamics
or Context optimization until the user explicitly starts training.

## Open decisions

1. Formal dataset sample count and train/validation split.
2. Held-out dynamics thresholds for one-step, short-horizon, and disagreement errors.
3. Dataset perturbation coverage and model/real rollout update ratio.
4. Safety-state projection or weights beyond full-observation tracking MSE.
5. Conditions needed to rule out a constant repair.
