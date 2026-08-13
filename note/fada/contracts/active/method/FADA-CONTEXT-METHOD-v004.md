---
contract_id: FADA-CONTEXT-METHOD-v004
status: active
effective_date: 2026-08-13
updated_date: 2026-08-13
supersedes: FADA-CONTEXT-METHOD-v003
prerequisite: FADA-METHOD-v005
scope: support-query inverse-dynamics supervision for one fixed left-knee actuator fault
---

# FADA Context Support-Query Method Contract

This contract records the human-selected method in Architecture 09 and Design Inspector 10. A
healthy-simulation Planner-IDM checkpoint is frozen. Two independent rollouts under the same command
and left-knee actuator-strength condition provide Support and Query. Only Context Encoder trains.

## First-stage condition

The first implementation fixes the left-knee actuator-strength multiplier to `0.7` and uses the
same straight-line command in Support and Query. This stage tests whether the frozen healthy IDM has
a usable latent calibration direction. It does not establish cross-fault identification or held-out
condition generalization.

## Data lifecycle

```text
healthy simulation -> train Planner P and Tracker/IDM I -> freeze P and I

fault simulation, rollout 1 -> Support S
fault simulation, independent rollout 2 -> Query Q

S = (target future from P, realized next state, executed action) over a complete rollout
Q = (realized history, realized future, executed action) from a disjoint rollout
```

Support and Query use the same straight command and the same `0.7` left-knee condition but independent
resets. Query action labels are the actions actually emitted by the frozen healthy Planner-IDM while
it runs closed-loop in the faulty simulator. The actuator-strength value is provenance and never a
Context input.

## Training path

The frozen IDM is factored at its existing final hidden representation:

```text
z_q = FrozenTrackerEncoder(query observation/action history, query realized future)
delta_z = ContextEncoder(Support)
a_hat_q = FrozenTrackerDecoder(z_q + broadcast(delta_z))
L_context = MSE(a_hat_q[first], a_exec_q[first])
```

`delta_z` has the healthy IDM hidden dimension and is constant for the whole Query rollout. It is
broadcast over the `K` latent tokens. Only the physically executed first action is supervised.
Tracker Encoder and Decoder parameters are frozen, but their operations remain in the autograd graph
so action loss reaches `delta_z` and Context Encoder.

The Planner also records the correct command-conditioned Query Intent. It does not replace the
realized future in the offline inverse-dynamics supervision row. At deployment, where future realized
states do not yet exist, the Planner Intent occupies the IDM future input.

## Deployment path

```text
first fault rollout -> Support -> one fixed delta_z
current fault history + command -> Frozen Planner -> Intent
current fault history + Intent -> Frozen Tracker Encoder -> z
Frozen Tracker Decoder(z + delta_z) -> complete action -> execute first action
```

No optimizer, Query label, future realized state, fault identity, or actuator-strength value is
available during deployment.

## Invariants

- Context reads only Support target motion, realized state, and executed action.
- Query labels and Query future never enter Context Encoder.
- Support and Query are disjoint rollouts under exactly the same command and condition.
- One Support produces one condition-level `delta_z`; it is not recomputed at control frequency.
- Fusion is exactly `z_repaired = z + delta_z` immediately before the frozen action head.
- `delta_z=0` is exactly equivalent to the original IDM forward path.
- The primary loss supervises only the actually executed first Query action.
- Planner, Tracker Encoder, Tracker Decoder, and normalizers remain unchanged.
- Evaluation compares a zero-Context fault rollout with a fixed-Context fault rollout; action MSE is
  training evidence, not closed-loop trajectory acceptance.

## Forbidden behavior

- Do not train or load a privileged teacher.
- Do not use fault strength, defect identity, Query action, or Query future as Context input.
- Do not supervise an invented `delta_z` target.
- Do not pass trajectory loss through a learned differentiable dynamics model.
- Do not train on an action chunk entry that was predicted but never executed.
- Do not claim condition generalization from the fixed-`0.7` first-stage experiment.

## Required evidence before training

- Exact zero-Context equivalence through the newly exposed IDM latent interface.
- Support/Query shape, finite, command-equality, condition-identity, and disjoint-pair validation.
- Query first-action labels originate from the fault rollout.
- Context optimizer owns exactly Context parameters; frozen checkpoint parameters do not change.
- Query action loss produces a finite nonzero Context gradient on a semantic fixture.
- A bounded real-MuJoCo preflight confirms the `0.7` configuration and emits the zero-Context Query
  action MSE before a long training run.

## Design-point and owner mapping

| Design point | Accepted decision | Implementation owner |
|---|---|---|
| `ICA-DP-01` | fixed left-knee `0.7`, hidden from Context | `mujoco_left_knee_070.yaml`, preflight fault validator |
| `ICA-DP-02` | frozen Planner records correct Intent | `FADAPlanner`, Support-Query collector |
| `ICA-DP-03` | frozen IDM exposes `z`, then decodes `z + delta_z` | `FADAInverseDynamicsModel`, `FrozenIDMSupportQueryPolicy` |
| `ICA-DP-04` | full Support target/realized/action sequence | `SupportContextBatch`, collector |
| `ICA-DP-05` | one complete Support emits one fixed `[B,128]` residual | `FADASupportContextEncoder` |
| `ICA-DP-07` | independent Query realized-future/first-action loss | `ContextQueryBatch`, `context_first_action_loss` |
| `ICA-DP-08` | trained frozen re-execution uses Planner Intent | `FrozenIDMSupportQueryPolicy.act_with_context` |
