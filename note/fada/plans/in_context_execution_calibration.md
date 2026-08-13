---
status: implementation-and-preflight-complete-training-not-started
updated_date: 2026-08-13
authority: FADA-CONTEXT-METHOD-v005
training_authority: FADA-CONTEXT-TRAIN-v004
scope: implement fixed-0.7 multi-window Support-Query Context action supervision
---

# In-Context Execution Calibration Plan

The differentiable-dynamics and single-anchor Query implementations remain stopped history. The
human owner accepted the Design Inspector 10 multi-sliding-window extension. This plan does not
authorize training.

## Accepted method

```text
sample hidden execution condition xi
-> collect Support (target motion, realized motion, executed action)
-> Context encodes Support once into fixed delta_z_xi
-> independently collect a complete disjoint no-Context Query under the same command and xi
-> construct every valid H=30, K=6 causal Query window
-> encode each window's realized future through the frozen Tracker latent interface
-> reuse the same Support delta_z for every window before the frozen action head
-> reconstruct each physically executed Query first action and average valid-window losses
-> update only Context Encoder
```

Support is Context input, not a repair label. Query actions are emitted by the healthy-trained frozen
Planner-IDM while running closed-loop in the fixed-`0.7` fault simulator. Fault identity remains
hidden from Context. Training uses Query realized futures; deployment uses Planner Intent. This
change targets temporal coverage only and does not guarantee healthy-trajectory recovery.

## Implementation steps

1. [completed] Replace the Query row with pair-owned `[P,W,...]` windows, anchors, and validity mask.
2. [completed] Fix complete Query length to `L=60` and construct all 26 causal windows.
3. [completed] Introduce dataset schema v2; prior single-anchor artifacts require explicit legacy loading.
4. [completed] Compute Context once per pair, reuse fixed `delta_z`, and apply masked first-action mean.
5. [completed] Add alignment, invalid-rollout, split-leakage, masking, freeze, and schema tests.
6. [completed] Run an 8-pair, 208-window, no-optimizer MuJoCo preflight.
7. Only after a new explicit authorization, train and compare single-anchor, multi-window, and zero-Context groups.

## Stop condition

Do not start long training if causal alignment or rollout split tests fail, if zero-Context all-window
MSE is non-finite/indistinguishable from zero, if Context gradients are zero/non-finite, if any
frozen parameter changes, or if Support/Query provenance fails validation.
