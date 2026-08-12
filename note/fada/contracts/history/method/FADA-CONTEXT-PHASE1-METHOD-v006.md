---
contract_id: FADA-CONTEXT-PHASE1-METHOD-v006
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FADA-CONTEXT-PHASE1-METHOD-v005
prerequisite: FADA-CONTEXT-METHOD-v001
scope: behavior-anchored fixed-left-knee-0.9 privileged full-action teacher
superseded_by: FADA-CONTEXT-METHOD-v002
---

# FADA Context Phase-1 Full-Action Teacher v006 Contract

v005 initialized exactly from the walking actor but every saved checkpoint had already collapsed to
the stationary shortcut. v006 repairs the actor-update boundary instead of adding another terminal
condition or increasing trajectory reward penalties.

## Policy boundary

```text
baseline: actor_observation -> frozen original walking SAC -> complete 29D action
teacher:  actor_observation + true 29D motor-strength g -> trained teacher -> complete 29D action
```

The teacher remains one privileged network whose public output is a complete 29D action. It does not
fuse an action residual at inference. During training only, a separately frozen copy of the original
walking actor supplies a no-gradient behavior anchor on the same actor observation.

## Actor objective

```text
L_actor = L_SAC + 10.0 * MSE(a_teacher(obs, g), stopgrad(a_nominal(obs)))
```

Only teacher parameters belong to the actor optimizer. The nominal anchor is excluded from optimizer
and checkpoint state, is reconstructed from the hash-bound original checkpoint, and must remain in
evaluation mode with `requires_grad=false`.

## Preserved environment and quality gate

The fixed left-knee index `3 = 0.9`, command `(0.4, 0.0, 0.0)`, reset-yaw trajectory penalties,
forward-progress termination, paired seeds, and conjunctive v005 quality thresholds remain unchanged.
Training completion or finite actions are not evidence that walking recovered.

## Required evidence

- Exact initialization still matches the original walking action for every tested `g`.
- The anchor loss is zero at initialization, positive after teacher perturbation, and cannot update
  the nominal actor.
- Formal configuration fails closed if anchor coefficient, learning rate, save interval, or iteration
  budget drifts.
- A bounded checkpoint sweep must first prove forward survival/progress before full paired quality is
  evaluated.

## Non-scope

Context Encoder training, latent repair, generalized actuator faults, reward-scale changes, action
residual teachers, and claims that fixed `0.9` is a realistic hardware fault remain outside scope.
