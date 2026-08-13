---
contract_id: FADA-CONTEXT-METHOD-v003
status: superseded-history
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FADA-CONTEXT-METHOD-v002
prerequisite: FADA-METHOD-v005
scope: differentiable-model trajectory training of rollout-conditioned latent repair
implementation_status: design-only
superseded_by: pending In-Context Execution Calibration decision
---

# FADA Context Differentiable-Trajectory Method Contract

This route was stopped on 2026-08-12 after all 10 real-MuJoCo candidate gates rejected Context
updates. It is retained as negative method history and is not current implementation authority.

This contract records the accepted method design, not an implementation or feasibility result. The
healthy-simulation Tracker Encoder `E` and Decoder `D` remain frozen. Under a simulated actuator
fault, a first rollout supplies deployable history to Context Encoder `C`; a second rollout applies
`delta_z = C(H)` before `D`. A learned differentiable fault-dynamics model carries second-rollout
trajectory loss gradients back to `C`. No action, trajectory, or optimized `delta_z` is treated as a
direct Context label.

## Accepted causal chain

```text
healthy simulation -> train E and D -> freeze E and D
same command + left-knee strength 0.9 -> fault probe rollout tau_probe
deployable tau_probe history H -> C(H) -> delta_z
same fault + paired initial condition -> D(E(x) + delta_z) -> differentiable dynamics ensemble
predicted adapted trajectory tau_hat_adapt -> compare with healthy reference tau_ref
trajectory loss -> backpropagate through dynamics, D, and E -> update only C
current C -> real MuJoCo fault rollout -> aggregate visited data -> refresh dynamics model
```

## FADA-CTX-DYN-DP-01 / HEALTHY-REFERENCE / Frozen Nominal Representation

`E` and `D` are trained only on the healthy simulated robot. Their checkpoint identity is fixed
before Context training. Healthy rollouts under the same command define `tau_ref`; they do not
provide action labels for the faulty robot.

## FADA-CTX-DYN-DP-02 / FAULT-PROBE / Causal Context Input

The first rollout uses the frozen nominal path on the left-knee-strength-`0.9` robot. `C` consumes
only deployment-visible causal history `H`, including accepted observations, executed actions,
commands, and reset/time boundaries. True motor strength and simulator-private state remain audit
metadata and never enter `C`.

## FADA-CTX-DYN-DP-03 / LATENT-REPAIR / Frozen Decoder Path

```text
delta_z = C(H)
z_repaired = E(x) + delta_z
a = D(z_repaired)
```

Only `C` is trainable. `C` outputs a latent residual, not an action, action residual, replacement
latent, fault parameter, or Decoder weight update. Zero repair exactly recovers the nominal path.

## FADA-CTX-DYN-DP-04 / FAULT-DYNAMICS / Differentiable Surrogate

An ensemble `F_phi` learns the faulty transition from causal MuJoCo tuples, preferably as state
increments. Its supervised objective includes one-step accuracy and short multi-step rollout
accuracy. Training data must include nominal-fault actions, latent/action perturbations, and current
Context visited states. After each dynamics update, `F_phi` is frozen during Context updates.

## FADA-CTX-DYN-DP-05 / PAIRED-SECOND-ROLLOUT / Model Unroll

The adapted rollout starts from a state paired with the reference/probe lifecycle and uses the same
fault and command. Inside the differentiable model:

```text
x_hat_0 = paired initial state
a_t = D(E(x_hat_t) + C(H))
x_hat_(t+1) = F_phi(x_hat_t, a_t)
```

The first accepted version emits one `delta_z` after a probe window and holds it fixed over a short
adaptation horizon. A time-varying residual is a later design decision.

## FADA-CTX-DYN-DP-06 / TRAJECTORY-LOSS / Context Gradient

The primary objective compares predicted adapted task trajectory with the healthy reference:

```text
L_C = L_track(tau_hat_adapt, tau_ref)
    + lambda_z * ||delta_z||^2
    + lambda_smooth * L_action_smooth
    + lambda_uncertainty * disagreement(F_1 ... F_M)
```

Task-space trajectory components may include base velocity, orientation, joint state, foot state,
and safety. Exact fields and weights remain open. Frozen parameters receive no optimizer updates,
but the computation graph must remain connected through `F_phi`, `D`, and `E` to `C`.

## FADA-CTX-DYN-DP-07 / REAL-SIM-AGGREGATION / Model-Bias Control

Context may exploit surrogate error. Therefore each bounded model-training round is followed by a
real MuJoCo rollout under the same fault. Model prediction and MuJoCo trajectory are compared, new
visited transitions enter the dynamics dataset, and the ensemble is refreshed before further
Context optimization. Surrogate-only improvement cannot accept Context quality.

## FADA-CTX-DYN-DP-08 / IDENTIFIABILITY / Deployment Proof

Training must include distinguishable conditions, initially healthy and multiple fault strengths,
commands, phases, or fault times. History mask/shuffle and constant-`delta_z` baselines must degrade
relative to correct history. Deployment contains only frozen `E`, `C`, and `D`; neither the learned
dynamics model nor fault identity is deployed.

## Parameter ownership

| Component | Dynamics-model training | Context training | Deployment |
|---|---|---|---|
| Tracker Encoder `E` | frozen | frozen, differentiable path | frozen |
| Decoder `D` | frozen | frozen, differentiable path | frozen |
| Context Encoder `C` | frozen | only trainable component | frozen |
| Dynamics ensemble `F_phi` | trainable | frozen, differentiable path | absent |
| MuJoCo fault simulator | data/validation owner | no direct gradient | absent |

## Forbidden behavior

- Do not fine-tune `E` or `D` during Context training.
- Do not use healthy or fault actions as cross-dynamics ground truth for `C`.
- Do not regress a searched or optimized `delta_z` as semantic truth.
- Do not claim `loss.backward()` crosses an ordinary MuJoCo rollout.
- Do not accept improvement measured only inside the learned dynamics model.
- Do not expose motor strength, simulator-private state, or `F_phi` to deployed Context inference.
- Do not infer history use from a single fixed fault, command, and initial state.

## Evidence required before implementation acceptance

- The left-knee `0.9` intervention and the exact command/reference lifecycle are runtime verified.
- A causal fault-transition dataset schema and paired rollout lifecycle pass contract tests.
- The dynamics ensemble passes held-out one-step, short-horizon, and uncertainty calibration gates.
- Context gradients are finite and only `C` parameters change.
- Correct-history Context beats zero repair, constant repair, and shuffled/masked history in MuJoCo.
- Model-predicted improvement transfers to paired MuJoCo rollouts without safety regression.
- Deployment strict-loads frozen `E/C/D` and excludes `F_phi` and privileged fields.

## Open decisions

1. Exact `E/D` checkpoint and latent dimension.
2. Probe history fields, length, and reset/session lifecycle.
3. Reference trajectory fields, alignment, horizon, weights, and safety thresholds.
4. Dynamics state representation, ensemble size, architecture, and uncertainty threshold.
5. Perturbation/data-coverage schedule and real-MuJoCo aggregation cadence.
6. `delta_z` bounds, normalization, and fixed-output horizon.
7. Fault/command/phase family needed to establish history dependence.
