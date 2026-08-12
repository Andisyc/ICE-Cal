---
date: 2026-08-12
branch: codex/context-differentiable-trajectory
contracts: FADA-CONTEXT-METHOD-v003, FADA-CONTEXT-TRAIN-v002
evidence_class: contract-confirmed and runtime-confirmed checkpoint sentinel
---

# Context Differentiable Core Evidence

## Implemented boundary

- `trajectory_context.py` maps causal observation/action history plus command to a bounded residual
  with the exact FADA Planner future shape, then injects it before the frozen IDM.
- `fault_dynamics.py` owns causal contiguous transition batches, state-increment ensemble prediction,
  one-step loss, and short multi-step loss.
- `differentiable_rollout.py` rolls one fixed Context residual through frozen Planner/IDM/dynamics
  parameters, updates deployable histories, and computes tracking, latent, smoothness, and ensemble
  disagreement loss components.
- No training entrypoint, MuJoCo collector, checkpoint writer, or deployment path is connected yet.

## Verification

```text
UV_CACHE_DIR=/tmp/fada_uv_cache uv run pytest -q
  tests/algos/test_fada_context_differentiable_trajectory.py
  tests/algos/test_fada_planner_idm.py
  tests/algos/test_fada_playback.py
  tests/algos/test_context_teacher_full_action_protocol.py
  tests/envs/locomotion/g1/test_actuator_strength.py
```

Observed: `32 passed`, with two existing Gymnasium cast warnings.

```text
UV_CACHE_DIR=/tmp/fada_uv_cache uv run pyright
  src/unilab/algos/torch/fada_context/trajectory_context.py
  src/unilab/algos/torch/fada_context/fault_dynamics.py
  src/unilab/algos/torch/fada_context/differentiable_rollout.py
```

Observed: `0 errors, 0 warnings, 0 informations`. Focused Ruff observed `All checks passed`.

The real `planner_idm_v005.pt` schema-2 sentinel emitted:

```text
delta_z=(1, 6, 98)
predicted_trajectory=(1, 3, 98)
actions=(1, 3, 29)
context_grad_l1=0.011846851732116193
planner_grad=false
idm_grad=false
dynamics_grad=false
finite=true
```

This proves the checkpoint-owned Planner output is the concrete Tracker latent boundary and that
loss gradients can pass through the frozen tensor path into Context only. It does not prove that a
learned fault dynamics model is accurate, that left-knee `0.9` degrades this checkpoint, or that
Context improves a real MuJoCo trajectory.

