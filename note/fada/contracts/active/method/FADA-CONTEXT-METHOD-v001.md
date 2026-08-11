---
contract_id: FADA-CONTEXT-METHOD-v001
status: active
effective_date: 2026-08-11
updated_date: 2026-08-11
prerequisite: FADA-METHOD-v005
scope: rollout-conditioned latent residual repair before a frozen Decoder
implementation_status: design-only
---

# FADA Context Method Contract

This contract is the semantic authority for the future Context Encoder. It records the method
confirmed by the human on 2026-08-11. It does not authorize Context implementation or training
before a valid anomalous condition and a qualified full-action privileged teacher exist.

## Three-stage ownership

### Stage 1: ideal Tracker and Decoder

Ideal information trains the Tracker Encoder `E` and Decoder `D`:

```text
ideal current input x -> Tracker Encoder E -> latent z -> Decoder D -> full 29D action
```

Formally:

```text
z = E(x)
a_nominal = D(z)
```

The resulting `E` and `D` are the nominal controller components reused in Context training.

### Stage 2: privileged full-action teacher

A separate privileged teacher `T` is trained under an accepted anomalous robot condition:

```text
current state/intent + privileged anomaly information g -> Teacher T -> full 29D teacher action
```

The teacher owns a complete repair-capable action policy. It is not an additive action-residual
branch and does not alter the nominal Tracker or Decoder. The exact anomaly owner and teacher input
schema remain open until a physical intervention demonstrably degrades the frozen nominal policy.

### Stage 3: Context distillation in latent space

During Context training, `E`, `D`, and `T` are frozen. Only Context Encoder `C` is updated:

```text
deployable rollout history H -> Context Encoder C -> latent residual delta_z
current ideal/deployable input x -> frozen Tracker Encoder E -> z
z_repaired = z + delta_z
frozen Decoder D(z_repaired) -> full 29D student action
```

The primary supervision is action-level teacher imitation:

```text
a_teacher = T(x, g)
a_student = D(E(x) + C(H))
L_context = distance(a_student, a_teacher)
```

The exact distance, weighting, regularization, and any optional auxiliary loss are not yet accepted.

## Frozen and trainable owners

| Component | Context-training state | Deployment state |
|---|---|---|
| Tracker Encoder `E` | frozen | frozen |
| Decoder `D` | frozen | frozen |
| Privileged Teacher `T` | frozen label provider | absent |
| Context Encoder `C` | trainable | frozen |

No optimizer, gradient update, LoRA update, or checkpoint replacement occurs during deployment.

## Required invariants

- `C(H)` is `delta_z`, not a complete action, an action residual, or a replacement latent.
- `delta_z` has exactly the shape and latent semantics required for addition to `z`.
- Fusion is exactly `z_repaired = z + delta_z`, immediately before the frozen Decoder.
- The Decoder interface and output remain the complete 29D action.
- The privileged teacher output is a complete 29D action.
- Zero Context repair recovers the nominal path exactly: `D(E(x) + 0) = D(E(x))`.
- Context training changes only Context Encoder parameters.
- Deployment consumes only causal, deployable observations and never consumes `g` or teacher output.

## Forbidden interpretations

- Do not fuse `delta_action` after the Decoder.
- Do not define `z_repaired = C(H)` or otherwise replace Tracker latent `z`.
- Do not update Tracker Encoder or Decoder weights while distilling Context.
- Do not expose `g`, simulator parameters, or teacher-only state to the deployed Context path.
- Do not claim Context uses rollout information merely because training succeeds on one fixed
  condition; a constant latent correction must be ruled out by the accepted evaluation design.
- Do not reinterpret the failed Phase-1 privileged residual-teacher experiments as the architecture
  of this contract.

## Evidence required before acceptance

- A demonstrably harmful but repairable anomaly is established against the frozen nominal policy.
- A privileged full-action teacher passes paired quality gates under that anomaly.
- Tracker Encoder and Decoder parameter identity is unchanged through Context training.
- The zero-`delta_z` path is numerically equivalent to the original nominal path.
- Context input fields, temporal window, reset/session boundary, and causality are contract-defined.
- The student action approaches the teacher action on held-out rollouts without privileged inputs.
- No-Context, zero-Context, history-shuffle, and distinguishable-condition evaluations rule out a
  constant correction and demonstrate use of rollout evidence.
- Deployment inference is verified with `E`, `D`, and `C` frozen and with `T` and `g` absent.

## Open decisions

1. Valid anomaly/intervention owner and its parameterization.
2. Privileged teacher architecture and exact privileged input schema.
3. Context rollout fields, history length, temporal alignment, and lifecycle.
4. Latent residual normalization, magnitude constraints, and optional regularization.
5. Exact action-level distillation loss and any accepted auxiliary loss.
6. Precise-trajectory evaluation object and thresholds.
7. Whether one inferred Context may be reused across commands or motions.

Until these are separately accepted, this contract fixes only the architecture and parameter
ownership stated above.
