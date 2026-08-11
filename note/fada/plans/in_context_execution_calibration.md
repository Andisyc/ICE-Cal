---
status: accepted-method-design
updated_date: 2026-08-11
authority: FADA-CONTEXT-METHOD-v001
scope: rollout-conditioned latent repair with frozen Tracker Encoder and Decoder
---

# In-Context Execution Calibration Plan

The active semantic authority is
`note/fada/contracts/active/method/FADA-CONTEXT-METHOD-v001.md`. This plan tracks the work needed to
make that architecture trainable and testable; it does not override the contract.

## Accepted architecture

### 1. Train the ideal controller representation

```text
ideal input -> Tracker Encoder E -> z -> Decoder D -> full 29D action
```

### 2. Train a privileged repair teacher

```text
state/intent + privileged anomaly information g -> Teacher T -> full 29D action
```

The teacher is an independent full-action oracle for the accepted anomalous condition. The failed
fixed-left-knee Kp/Kd residual-teacher experiments are retained as negative evidence and do not
define this architecture.

### 3. Distill rollout information into latent repair

```text
rollout history H -> Context Encoder C -> delta_z
current input x -> frozen E -> z
z_repaired = z + delta_z
frozen D(z_repaired) -> full 29D student action
```

Freeze `E`, `D`, and `T`; train only `C`. The primary objective compares the decoded student action
with the teacher's complete action:

```text
distance(D(E(x) + C(H)), T(x, g))
```

At deployment, `g` and `T` are absent and all model weights remain frozen.

## Design register

| Owner | Accepted role | Current status |
|---|---|---|
| Tracker Encoder `E` | maps current ideal/deployable input to nominal latent `z` | architecture accepted; checkpoint owner to identify |
| Decoder `D` | maps nominal or repaired latent to complete 29D action | architecture accepted; checkpoint owner to identify |
| Teacher `T` | privileged full-action oracle under a valid anomaly | unavailable; previous intervention invalid |
| Context Encoder `C` | maps causal rollout history to `delta_z` | semantic output accepted; architecture/input open |
| Latent fusion | exactly `z_repaired = z + delta_z` before `D` | accepted |

## Work sequence

1. Select a physically meaningful intervention that measurably degrades the frozen nominal policy.
2. Define the privileged full-action teacher contract and paired quality gate for that intervention.
3. Train and accept the teacher before creating Context labels.
4. Identify the exact pretrained Tracker Encoder and Decoder checkpoints and verify their nominal
   interface, latent dimension, and zero-repair behavior.
5. Contract the deployable rollout fields, temporal window, alignment, and episode/session boundary.
6. Implement Context Encoder and latent addition while freezing Tracker Encoder and Decoder.
7. Distill against the teacher's complete action and verify parameter ownership.
8. Evaluate held-out imitation, trajectory precision, privilege exclusion, and history dependence.

## Required controls

- original nominal path `D(E(x))`;
- zero latent repair `D(E(x) + 0)`;
- learned Context repair;
- history-shuffled or history-masked Context;
- at least two distinguishable execution conditions so a constant `delta_z` cannot satisfy the
  scientific claim;
- privileged full-action teacher as an upper-bound reference, not a deployment input.

## Open decisions

1. The valid anomaly and simulation owner.
2. Teacher policy architecture, observations, reward, and quality thresholds.
3. Exact Tracker/Decoder checkpoint and the latent dimension of `z`.
4. Context history contents and temporal window.
5. Latent scaling, bounds, normalization, and regularization.
6. Exact action-level loss and any optional auxiliary loss.
7. Calibration lifecycle and cross-command reuse.
8. Precise trajectory target and publication-level acceptance metrics.

## Current stop condition

Do not implement or train Context Encoder yet. The next safe boundary is to select and baseline a
valid intervention, then train a privileged teacher that outputs complete 29D actions and passes a
paired quality gate. Any anomaly-owner choice remains a new human decision.

## Novelty status

History-conditioned latent adaptation with frozen policy weights is not novel by itself. The
research claim remains unconfirmed until the method demonstrates target-specific precision from a
bounded rollout, proves that the latent repair uses rollout evidence rather than a constant offset,
and compares against nominal, history-conditioned, privileged-teacher, and optimization-based
adaptation baselines under the same target-data budget.
