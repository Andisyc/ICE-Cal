---
contract_id: FADA-CONTEXT-METHOD-v002
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FADA-CONTEXT-METHOD-v001
prerequisite: FADA-METHOD-v005
scope: direct supervised rollout-conditioned latent repair before a frozen Decoder
implementation_status: design-only
superseded_by: FADA-CONTEXT-METHOD-v003
---

# FADA Context Direct-Supervision Method Contract

This contract records the human decision to remove the trained privileged-teacher stage. A
fault-condition trajectory-tracking expert generates same-state complete-action labels directly;
only the Context Encoder is trained against those labels. This contract does not authorize model
implementation or training before the fault, label, and Decoder-reachability gates are accepted.

## Accepted causal chain

```text
define fault and reference
  -> prove the frozen nominal policy is measurably degraded
  -> query a trajectory-tracking expert in the same fault dynamics and visited state
  -> pair deployable causal history with the expert complete-action label
  -> verify the frozen Decoder can express the required corrective actions
  -> train only Context Encoder
  -> evaluate privilege-free execution
```

There is no trained privileged teacher model between the trajectory-tracking expert and Context
supervision.

## Owners and interfaces

### Nominal action path

```text
current deployable input x -> frozen Tracker Encoder E -> latent z
z -> frozen Decoder D -> complete 29D nominal action
```

### Offline label owner

```text
FaultExpert(same visited state, reference, privileged fault state) -> complete 29D action a_expert
```

The expert is an offline label generator. Its exact provider may be MPC, trajectory optimization,
or another explicitly accepted tracking solver. Normal-dynamics actions are not repair labels.

### Context training path

```text
deployable causal rollout history H -> Context Encoder C -> latent residual delta_z
z_repaired = E(x) + C(H)
a_context = D(z_repaired)
```

Only `C` is trainable. `E` and `D` remain frozen.

## Direct supervised objective

The primary target is the expert's complete corrective action, not `delta_z` and not a future state:

```text
L_action = distance(D(E(x) + C(H)), a_expert)
```

Optional terms may regularize `delta_z`, enforce temporal consistency, or predict future state as an
auxiliary task. Future state or trajectory is an evaluation object and optional auxiliary target;
it does not replace complete-action supervision.

Direct `delta_z` labels are forbidden as the primary target because the frozen Decoder may map
multiple latent residuals to the same action. A per-sample optimized latent may be used only as a
reachability diagnostic, not silently promoted to semantic ground truth.

## Dataset and privilege boundary

Each supervised row must bind:

```text
(causal history H, current input x, reference identity, fault identity for provenance,
 same-state complete expert action a_expert)
```

- `H` and `x` contain only deployment-visible fields.
- Fault identity and privileged simulator state are available to the offline expert and dataset
  audit only; they are unreachable from `C`'s forward input.
- The expert action is generated from the same simulator snapshot, fault dynamics, reference, and
  control time as the student input.
- Data must cover nominal/fault baseline states and Context-policy visited states so supervised
  learning does not rely only on expert-state support.

## Decoder reachability gate

Before Context training, bounded per-sample optimization checks whether a free latent residual can
express the expert action through the frozen Decoder:

```text
min_delta_z distance(D(E(x) + delta_z), a_expert) + lambda * ||delta_z||^2
```

This probe does not create the training target. If the required corrective actions remain outside
the frozen Decoder's reachable action manifold, stop and return to the human design boundary.

## Parameter ownership

| Component | Context-training state | Deployment state |
|---|---|---|
| Tracker Encoder `E` | frozen | frozen |
| Decoder `D` | frozen | frozen |
| Context Encoder `C` | trainable | frozen |
| Trajectory-tracking expert | offline label generator | absent |

No optimizer, gradient update, LoRA update, or checkpoint replacement occurs during deployment.

## Invariants

- `C(H)` is `delta_z`, not a complete action, action residual, or replacement latent.
- Fusion is exactly `z_repaired = z + delta_z` immediately before the frozen Decoder.
- The Decoder output remains the complete 29D action.
- Context training changes only Context Encoder parameters.
- Zero Context repair exactly recovers the nominal path.
- The primary supervised label is a same-state complete corrective action.
- Deployment consumes causal deployable history and never consumes fault parameters, expert state,
  or expert output.

## Forbidden behavior

- Do not train or insert a privileged-teacher network as an intermediate label provider.
- Do not use normal-dynamics policy actions as fault-correction labels.
- Do not directly supervise an arbitrary `delta_z` as if it were unique ground truth.
- Do not use future-state prediction alone to claim control supervision.
- Do not update Tracker Encoder or Decoder during Context training.
- Do not continue after the fault-validity, expert-quality, or Decoder-reachability gate fails.

## Design-point mapping

| Design point | Contract anchor |
|---|---|
| `FADA-CTX-DP-01` | fault validity before label collection |
| `FADA-CTX-DP-02` | shared reference trajectory and tolerance contract |
| `FADA-CTX-DP-03` | same-state fault tracking expert |
| `FADA-CTX-DP-04` | direct complete-action supervision without a teacher model |
| `FADA-CTX-DP-05` | causal deployable rollout history |
| `FADA-CTX-DP-06` | frozen latent-repair interface and Decoder reachability |
| `FADA-CTX-DP-07` | privilege lifecycle and deployment exclusion |
| `FADA-CTX-DP-08` | conjunctive paired quality gates |

## Evidence required before acceptance

- A harmful but repairable fault is established against the frozen nominal policy.
- The trajectory-tracking expert passes the shared reference and safety gate under that fault.
- Dataset rows prove same-state label alignment and Context-input privilege exclusion.
- Free-latent probes show the frozen Decoder can express the required corrective actions.
- Tracker Encoder and Decoder identities remain unchanged through Context training.
- Held-out action imitation and closed-loop trajectory quality pass separately.
- History-shuffle/mask and distinguishable-condition tests rule out a constant `delta_z`.
- Deployment inference is verified with only `E`, `C`, and `D` present and frozen.

## Open decisions

1. Fault family, parameterization, and held-out split.
2. Reference trajectory object and conjunctive tolerance thresholds.
3. Tracking-expert provider and same-state query lifecycle.
4. Exact Tracker/Decoder checkpoint, latent dimension, and reachability threshold.
5. Context rollout fields, history length, temporal alignment, and lifecycle.
6. Action distance, latent regularization, temporal consistency, and optional future auxiliary loss.
7. Dataset aggregation policy for Context-policy visited states.
8. Cross-command and cross-fault reuse of one inferred Context.
