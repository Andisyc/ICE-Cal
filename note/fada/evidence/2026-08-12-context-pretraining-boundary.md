---
date: 2026-08-12
branch: codex/context-differentiable-trajectory
contracts: FADA-CONTEXT-METHOD-v003, FADA-CONTEXT-TRAIN-v002
evidence_class: implementation and bounded real-MuJoCo preflight
---

# Context Pretraining Boundary

The repository now owns the complete boundary immediately before optimization:

- `trajectory_data.py` persists paired fault history, healthy reference, and contiguous fault
  transitions under an architecture-bound schema.
- `trajectory_collector.py` restores one healthy MuJoCo snapshot into a same-model fault environment.
  MuJoCo rollout snapshots carry physics and sensor state but not model actuator gains, so the fault
  branch starts aligned while retaining the configured left-knee `0.9` gain.
- `training_setup.py` constructs frozen Planner/IDM, trainable Context and dynamics, and two disjoint
  optimizers.
- `preflight_fada_context_differentiable.py` collects and round-trips a bounded dataset, computes both
  losses and gradients, and rejects any config that permits optimizer steps.

## Real checkpoint and MuJoCo probe

Checkpoint: `/Users/sss9999/locomotion/FADA/planner_idm_v005.pt`

SHA-256: `d35a32d93b0387e534f6fcdd86b724c44187e308dbca1412435bffe95b6ed90c`

Observed with one environment and one accepted pair:

```text
observation_history=(1, 30, 98)
healthy_reference=(1, 3, 98)
fault_state=(1, 34, 98)
fault_action=(1, 33, 29)
rejected_done_samples=0
metadata_round_trip=true
dynamics_grad_l1=48.85814619064331
context_grad_l1=0.04238511750008911
planner_unchanged=true
idm_unchanged=true
context_unchanged=true
dynamics_unchanged=true
optimizer_steps=0
training_started=false
```

This is readiness evidence, not dynamics-quality or Context-improvement evidence. Formal training has
not started.
