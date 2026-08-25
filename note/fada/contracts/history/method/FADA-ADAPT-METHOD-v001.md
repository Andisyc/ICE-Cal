# FADA-ADAPT-METHOD-v001

Status: superseded for paper Figure 3(d) reproduction.

Superseded on 2026-08-21 by `FADA-ADAPT-METHOD-v002`, which forbids action/command leakage through
the future-state input.

## Scientific owner

This contract owns only the paper's few-shot target adaptation stage. It consumes the frozen
Planner-IDM source checkpoint and the Stage-C ordinary target artifact. It is independent from
`FADA-CONTEXT-METHOD-v003`.

The paper-defined method is:

```text
W_target = (O_history^H, A_history^H, Y_executed^K, U_executed^K)
policy_ft = (P_phi, I_{psi + Delta_psi})
loss = MSE(Pi_1(I_{psi + Delta_psi}(O_history, A_history, Y_executed)),
           Pi_1(U_executed))
```

- freeze Planner `P_phi`;
- freeze pretrained IDM weights `psi`;
- optimize only IDM LoRA parameters `Delta_psi`;
- use rank `8`, alpha `16`, and adapter-input dropout `0.05`;
- use target executed future/action pairs from the same rollout segment;
- supervise only the first action and deploy with the same receding-horizon first action.

Forbidden inputs are target rewards, Oracle or teacher labels, privileged simulator state, source
replay, fault coefficients, and Context-repair labels.

## Repository-owned injection decision

The paper does not name PyTorch module paths for LoRA injection. This repository therefore owns one
explicit, persisted injection manifest: every directly invoked `nn.Linear` transform inside the IDM
(input embeddings, Transformer feed-forward projections, and action head). PyTorch
`MultiheadAttention.out_proj` children are excluded because its functional forward bypasses the
child module call; registering dead adapters would violate the gradient/update contract. Attention
Q/K/V weights remain frozen.

The exact ordered target-module list is stored in every adapted checkpoint. A changed or missing
list is a schema incompatibility, not a silent fallback.

## Preserved boundary

The source checkpoint reader, source training, Stage-C collection, Planner command interface,
observation/action dimensions, playback history lifecycle, and first-action execution remain
unchanged. Zero-initialized LoRA B matrices make the injected policy exactly equal to the source
policy before the first update.

## Evidence ceiling

Module evidence may prove shapes, roles, freeze/detach/update ownership, exact zero-delta behavior,
first-action loss, persistence, and offline playback loading. It cannot prove that a schedule learns
a useful correction, improves simulation, transfers to hardware, or is safe to deploy.

Paper authority: FADA Section 4.2, Equations 4.4-4.6, Algorithm 1 T1-T3, and Appendix B.3,
`https://arxiv.org/html/2606.28476v1`.
