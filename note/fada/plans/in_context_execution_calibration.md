---
status: implementation-active
updated_date: 2026-08-13
authority: FADA-CONTEXT-METHOD-v004
training_authority: FADA-CONTEXT-TRAIN-v003
scope: implement fixed-0.7 Support-Query Context action supervision
---

# In-Context Execution Calibration Plan

The differentiable-dynamics implementation remains stopped history. The human owner accepted Design
Inspector 10 Support-Query action supervision and authorized implementation.

## Accepted method

```text
sample hidden execution condition xi
-> collect Support (target motion, realized motion, executed action)
-> Context encodes Support once into fixed delta_z_xi
-> independently collect a disjoint no-Context Query under the same command and xi
-> encode Query realized future through frozen IDM latent interface
-> add Support delta_z immediately before the frozen action head
-> reconstruct the physically executed Query first action
-> update only Context Encoder
```

Support is Context input, not a repair label. Query action is emitted by the healthy-trained frozen
Planner-IDM while running closed-loop in the fixed-`0.7` fault simulator. Fault identity remains
hidden from Context. Training uses Query realized future; deployment uses Planner Intent.

## Implementation steps

1. Expose the existing IDM final hidden tokens as a checkpoint-compatible latent interface.
2. Add a full-Support Context Encoder and exact `z + delta_z` frozen-IDM wrapper.
3. Define a versioned Support/Query dataset with strict provenance and causal first-action labels.
4. Collect two independent fixed-`0.7` rollouts per pair under one straight command.
5. Add preflight, offline trainer, checkpoint persistence, and focused contract tests.
6. Run bounded real-MuJoCo collection/backward evidence before long training.

## Stop condition

Do not start long training if zero-Context Query action MSE is non-finite/indistinguishable from zero,
if Context gradients are zero/non-finite, if any frozen parameter changes, or if Support/Query
provenance fails validation.
