---
contract_id: FADA-CONTEXT-METHOD-v005
status: historical
effective_date: 2026-08-13
updated_date: 2026-08-13
superseded_date: 2026-08-14
superseded_by: FADA-CONTEXT-METHOD-v006
supersedes: FADA-CONTEXT-METHOD-v004
prerequisite: FADA-METHOD-v005
scope: fixed-delta Support-Query calibration with full-Query multi-window supervision
implementation_status: implemented-preflight-passed-training-not-started
---

# FADA Context Multi-Window Support-Query Method Contract

This contract records the human-selected **FADA-style multi-sliding-window Context training**
extension to Architecture 09 and Design Inspector 10. It preserves the Support-derived fixed
`delta_z` method and changes only the temporal coverage of Query supervision.

## Scope and non-scope

The first stage fixes left-knee actuator strength to `0.7`, command to `[0.4, 0, 0]`, history length
to `H=30`, and future horizon to `K=6`. The healthy Planner, Tracker Encoder, and Tracker Decoder are
strictly frozen. Only Context Encoder trains.

This version addresses the previous dataset's one-anchor-per-Query limitation. Because its label is
still the action emitted by the original frozen policy in the fault simulator, improved action
reconstruction does not establish recovery toward the healthy trajectory.

State-dependent `delta_z_t`, privileged fault inputs, teacher actions, differentiable dynamics, and
trajectory-gradient training are outside this version.

## Data lifecycle

```text
fault rollout 1, no Context -> complete Support S
fault rollout 2, no Context -> complete independent Query trajectory Q

delta_z = ContextEncoder(S)
Q -> all causally valid sliding windows q_t
same delta_z is reused for every q_t from Q
```

Support and Query share the exact command and fault condition but have distinct reset/rollout
identities. Dataset splitting happens by complete rollout group before any training sampler sees a
window. Windows from one Query must never cross the train/validation boundary.

## Pair-window representation

The semantic batch axes are `P` Support-Query pairs and `W` Query windows per pair:

```text
Support target future       [P, S, K, O]
Support realized state      [P, S, O]
Support executed action     [P, S, A]

Query observation history  [P, W, H, O]
Query action history       [P, W, H, A]
Query realized future      [P, W, K, O]
Query executed first action [P, W, A]
Query window anchor/mask    [P, W]
```

For a fixed-length Query rollout of `L` transitions, valid anchors satisfy
`H - 1 <= t <= L - K`, giving `W = L - H - K + 2`. A mask may represent different valid window
counts, but padded windows never enter the loss.

## Causal window contract

For transition `i`, the collector records state `s_i`, previous action `a_(i-1)`, executed action
`a_i`, and resulting next state `s_(i+1)`. Each window anchored at `t` is exactly:

```text
observation history = s_(t-H+1) ... s_t
action history      = a_(t-H)   ... a_(t-1)
realized future     = s_(t+1)   ... s_(t+K)
first-action label  = a_t
```

No window may cross reset, termination, truncation, or command change. Only `a_t`, the physical
action that produced the first realized future state `s_(t+1)`, is supervised.

## Training path

```text
delta_z_p = ContextEncoder(S_p)
z_(p,t) = FrozenTrackerEncoder(history_(p,t), realized_future_(p,t))
A_hat_(p,t) = FrozenTrackerDecoder(z_(p,t) + delta_z_p)
L_context = masked_mean_(p,t) MSE(A_hat_(p,t)[0], a_fault_(p,t))
```

Context Encoder runs once per sampled Support. Its `[P,D]` output is broadcast only across the
windows belonging to the same pair and across the `K` Tracker latent tokens. The five nonexecuted
action-chunk entries remain outside the loss.

## Deployment path

Deployment is unchanged:

```text
first fault rollout -> Support -> one fixed delta_z
current history + command -> Frozen Planner -> six-step Intent
history + Intent -> Frozen Tracker Encoder -> z_t
Frozen Tracker Decoder(z_t + delta_z) -> execute first action only
next control step -> replan
```

## Invariants and forbidden behavior

- One complete Support produces one fixed condition-level `delta_z` for the complete Query.
- Query realized futures and labels never enter Context Encoder.
- Planner and complete Tracker remain frozen and outside the optimizer.
- Training and validation are disjoint by Support/Query rollout group, not by window.
- Loss is the mean over valid executed first actions only.
- `delta_z=0` remains exactly the healthy-trained Tracker path.
- Do not flatten windows in a way that loses pair ownership or changes pair weighting silently.
- Do not claim healthy-trajectory repair from action-MSE improvement.

## Required evidence before training

- Exact pair-window shapes, identities, masks, and serialization round trip.
- A numbered-transition fixture proving history/future/action causal alignment at every anchor.
- A negative test rejecting reset, done, truncation, or command-changing windows.
- One Support forward per pair and exact fixed-`delta_z` reuse across its valid windows.
- Masked mean ignores padded windows and all five nonexecuted action-chunk entries.
- Rollout-group split prevents every form of window leakage.
- Optimizer identity and parameter snapshots prove only Context changes.
- A bounded MuJoCo preflight reports all-window zero-Context action MSE before training.

## Design-point and owner mapping

| Design point | Accepted decision | Implementation owner/status |
|---|---|---|
| `ICA-DP-01` | fixed left-knee `0.7`, hidden from Context | fault config/validator; existing |
| `ICA-DP-02` | frozen Planner records Intent | Planner and collector; existing |
| `ICA-DP-03` | frozen Tracker decodes `z + delta_z` | Tracker latent interface; existing |
| `ICA-DP-04` | full Support target/realized/action sequence | Support data owner; existing |
| `ICA-DP-05` | one complete Support emits one fixed `delta_z` | Context Encoder; existing |
| `ICA-DP-06` | complete Query becomes causally aligned sliding windows | Query data/collector; implemented and preflight-passed |
| `ICA-DP-07` | masked all-window executed-first-action loss | loss/trainer; implemented and preflight-passed |
| `ICA-DP-08` | frozen receding-horizon deployment uses Planner Intent | playback/evaluator; unchanged |

## Fixed first-stage implementation choice

The first-stage config fixes Query rollout length to `L=60`. With `H=30` and `K=6`, every accepted
Query produces `W=26` anchors numbered `29..54`. This choice affects sample count and compute, not
the accepted causal semantics.

The bounded 2026-08-13 MuJoCo preflight accepted 8 pairs and 208 windows, measured zero-Context
all-window first-action MSE `2.3850443540140986e-05`, obtained Context gradient norm
`1.553474721731618e-04`, and performed zero optimizer steps. Formal training remains unstarted.

## Superseded-route evidence

The reason for replacing the one-anchor data construction is recorded in
`note/fada/evidence/2026-08-13-context-single-anchor-closed-loop.md`. This negative result motivates
the new coverage experiment but does not count as evidence that multi-window training will succeed.
