---
date: 2026-08-12
evidence_class: source-confirmed-literature
status: concept-research-only
scope: related work and feasibility of trajectory-scored latent search followed by Context amortization
authority: none; this note does not activate a training contract
---

# Context Route 2 Related Work and Feasibility

## Candidate method being compared

The proposal keeps the normal-dynamics Tracker Encoder `E` and Decoder `D` frozen. Under a motor
fault, a deployable causal history `H` is collected and a same-snapshot black-box optimizer searches
for a fixed latent residual `delta_z` that maximizes closed-loop reference-trajectory quality:

```text
trajectory/reference objective
  -> same-snapshot search over free delta_z
  -> validated delta_z* or high-quality elite set
  -> supervised amortization C(H) -> delta_z
  -> optional Decoder-aware action auxiliary
  -> held-out closed-loop trajectory acceptance
```

The supervision hierarchy is deliberate:

- trajectory quality is the root task objective and final acceptance oracle;
- `delta_z*` or an elite latent set is the main offline proxy target because ordinary MuJoCo
  rollouts do not expose the required cross-time gradient to the Context optimizer;
- action matching is optional Decoder-aware regularization, not proof of trajectory equivalence;
- motor strength is simulation provenance or an optional identification auxiliary, not deployment
  input.

## Closest primary-source precedents

| Work | Established component | Important difference from Route 2 |
|---|---|---|
| [Guided Policy Search](https://proceedings.mlr.press/v28/levine13.html) and [Learning Complex Neural Network Policies with Trajectory Optimization](https://proceedings.mlr.press/v32/levine14.html) | Trajectory optimization can generate guiding behavior that is amortized into a neural policy; the 2014 method alternates trajectory-cost optimization and policy matching. | These methods optimize trajectory distributions/actions and the policy together, rather than search one residual in a frozen nominal Decoder latent. |
| [RMA](https://arxiv.org/abs/2107.04034) | A recent-history adaptation module can provide a rapidly changing latent/context to a locomotion base policy. | The base policy is trained for varied environments together with a useful environment representation; Route 2 asks a normal-only frozen `E/D` to contain unseen fault compensation directions. |
| [UP-OSI](https://arxiv.org/abs/1702.02453) | Recent state/action history can identify changing dynamics parameters and condition control. | Its universal policy is explicitly trained across dynamic models and consumes the identified model parameters; Route 2 searches a task-optimal latent correction instead of physical parameters. |
| [Contextual policy search / C-REPS formulation](https://proceedings.mlr.press/v100/klink20a/klink20a.pdf) | Contextual RL can learn a conditional distribution over policy parameters `pi(theta|c)` from return `R(theta,c)`. | Route 2's context is a partially observed history, `theta` is a residual inside a frozen Decoder, and label generation is separated from supervised amortization. |
| [Learning to Adapt in Dynamic, Real-World Environments](https://arxiv.org/abs/1803.11347) | Recent experience can drive online adaptation under crippled body parts and support target-trajectory following. | It adapts a meta-trained dynamics model for model-based control instead of predicting a frozen-policy latent residual. |
| [BayesSim](https://arxiv.org/abs/1906.01728) | Black-box simulated trajectories can train likelihood-free inference of latent dynamics parameters. | It targets a posterior over physical simulator parameters; Route 2 targets a task-optimal control correction that need not identify the true fault. |
| [AdaptSim](https://proceedings.mlr.press/v229/ren23b.html) | Adaptation can be task-driven: matching physical parameters is not always necessary if the goal is target task performance. | It updates simulator-parameter distributions for policy retraining rather than directly searching a control latent. |
| [Fault-tolerant quadruped ACDR](https://arxiv.org/abs/2111.10005) | Motor/actuator-failure locomotion can be learned with RL and dynamics randomization. | It trains a robust policy across failure conditions; it does not preserve a normal-only frozen base and separately amortize latent search. |

No exact primary-source match for the complete combination was found in the focused search. That is
not a novelty proof; it is a bounded literature-search result.

## Feasibility assessment

Verdict: **plausible but conditional; the highest-risk assumption is frozen-Decoder reachability.**

### Strongly supported components

1. Recent proprioceptive state/action history can contain information about changing dynamics.
2. Closed-loop trajectory optimization can guide supervised function approximation.
3. Task-optimal adaptation need not recover the true physical fault parameter.
4. Offline expensive optimization can in principle be amortized into a fast neural forward pass.

### Unproven components specific to this proposal

1. A Decoder trained only under normal motor strength may not expose any latent direction that
   produces the actions required under actuator loss.
2. A single fixed `delta_z` over a horizon may be too weak; success may require state-dependent or
   phase-dependent correction.
3. Black-box search scales poorly with latent dimension. Direct CEM in a high-dimensional Tracker
   latent may be computationally infeasible or return unstable labels.
4. Passive history may not identify the fault. Different faults can yield similar short histories
   but require different corrections unless probing excitation is sufficient.
5. Good latent solutions may be multimodal. Point regression can average incompatible solutions.
6. A scalar trajectory score can be exploited; progress, lateral/yaw tracking, stability, saturation,
   and safety require conjunctive reporting rather than compensation inside one scalar alone.

## Smallest falsifying experiment

Do not implement Context Encoder first. For one accepted fault, one reference, and a small paired
snapshot set:

1. freeze the exact normal `E/D` checkpoint;
2. search a low-dimensional residual coordinate `u`, with `delta_z = B u`, before attempting the
   full latent dimension;
3. compare `delta_z=0` with searched `delta_z*` from identical snapshots using conjunctive trajectory
   metrics;
4. repeat search seeds to measure solution and score stability;
5. test whether one fixed residual survives the full evaluation horizon;
6. only if search improvement is repeatable, collect histories and test whether a held-out predictor
   can preserve that improvement.

Interpretation:

- search fails: reject or change the frozen latent interface before Context training;
- search succeeds but repeated solutions are unstable: retain elite sets/distributions or impose a
  canonical low-dimensional residual basis;
- stable search succeeds but history prediction fails: change probing/history identifiability;
- offline prediction succeeds but closed loop fails: repair visited-state coverage or move to
  trajectory-aware/RL fine-tuning.

## Literature-search boundary

Searches were restricted to primary paper pages and proceedings for rapid motor adaptation, online
system identification, contextual policy search, trajectory-optimization-guided policy learning,
task-driven adaptation, black-box trajectory inference, meta-adaptation, and actuator-failure
locomotion. This note makes no exhaustive novelty or publication claim.
