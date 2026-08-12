---
contract_id: FADA-CONTEXT-TRAIN-v002
status: active
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FADA-CONTEXT-TRAIN-v001
method_contract: FADA-CONTEXT-METHOD-v003
scope: differentiable fault-model trajectory optimization of Context Encoder only
implementation_status: design-only
---

# FADA Context Differentiable-Trajectory Training Contract

## Training stages

1. Strict-load and freeze healthy-simulation `E/D`; materialize paired healthy references.
2. Collect `(x_t, a_t, x_(t+1))` and lifecycle metadata from left-knee-`0.9` MuJoCo rollouts,
   including Decoder-reachable perturbations and current-Context visited states.
3. Train a fault-dynamics ensemble with one-step and scheduled short-horizon losses; freeze it.
4. Collect a causal probe history, compute one `delta_z=C(H)`, and unroll the paired adapted
   trajectory through the frozen ensemble, `E`, and `D`.
5. Backpropagate reference-trajectory, latent, smoothness, and uncertainty losses; the optimizer owns
   only `C.parameters()`.
6. Validate the updated Context in paired real MuJoCo rollouts, aggregate newly visited transitions,
   and refresh the ensemble before the next bounded round.

## Gradient contract

`E`, `D`, and `F_phi` have frozen parameters during a Context step but must not be detached. A test
must prove finite nonzero gradients reach `C`, and exact parameter snapshots must prove only `C`
changes. MuJoCo observations are data boundaries and are never claimed to retain autograd history.

## Admission gates

- The nominal and left-knee-`0.9` simulator interventions are runtime identified.
- Healthy reference and fault probe use the same command and accepted alignment lifecycle.
- Dataset rows preserve episode/reset/time/fault provenance while `C` inputs exclude privileged data.
- Dynamics ensemble accuracy and disagreement pass predeclared held-out short-horizon thresholds.
- Zero repair is numerically identical to the frozen nominal path.

## Training and evaluation evidence

- Report dynamics one-step and multi-step error separately from Context trajectory loss.
- Report surrogate and real-MuJoCo adapted trajectories separately; never merge their metrics.
- Compare zero repair, learned Context, constant repair, history mask, and history shuffle.
- Evaluate held-out faults/commands/phases separately from training conditions.
- Bind checkpoints to `E/D`, Context architecture, dynamics dataset, ensemble, and reference identities.
- Reject non-finite gradients, ensemble disagreement beyond threshold, falls, or paired lifecycle drift.

Exact architectures, horizons, loss weights, thresholds, commands, and runtime owners remain open.
This design-only contract authorizes documentation of the method, not implementation or training.
