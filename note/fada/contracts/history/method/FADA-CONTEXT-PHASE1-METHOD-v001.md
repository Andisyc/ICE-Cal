---
contract_id: FADA-CONTEXT-PHASE1-METHOD-v001
status: superseded
effective_date: 2026-08-10
updated_date: 2026-08-10
superseded_by: FADA-CONTEXT-PHASE1-METHOD-v002
prerequisite: FADA-METHOD-v005
scope: privileged residual-teacher feasibility under controlled 29D actuator-strength variation
---

# FADA Context Phase-1 Method Contract

This contract governs only the first feasibility experiment for execution repair. It does not claim
that a rollout-conditioned Context Encoder is trained, identifiable, or deployment-ready.

The simulator owns a true 29D actuator-strength vector `g`. The vector changes reset-time position
servo gains and is also exposed to the privileged teacher. The frozen walking SAC checkpoint owns
the nominal action. The teacher may output only a bounded additive residual:

```text
nominal_action = frozen_walking_sac(actor_observation)
delta_action = privileged_residual_teacher(actor_observation, g)
executed_action = clip(nominal_action + delta_action, -1, 1)
```

The teacher must never replace or update the nominal walking policy. Zero deterministic residual must
reproduce the nominal actor exactly. Training, collection, and playback must use the same actor-owned
fusion rule.

Phase-1 training must vary `g` across resets. At minimum its support contains nominal strength, a
left-knee anomaly, and a right-knee anomaly. A run with only one fixed left-knee value is a playback
probe and cannot establish conditional repair because the residual network could memorize one case.

The future Context Encoder does not receive `g` and does not regress `g`. Its future output is a
temporary latent execution Context inferred from deployable rollout history. Distillation of that
latent path is outside this contract.

## Paired feasibility measurement

Teacher feasibility is measured against the frozen nominal actor from the exact same initialized
environment snapshot. Both branches receive the same `g`, command, physics state, observation,
history carrier, and RNG state. Autoreset is disabled during each branch. A physical termination is
a fall; a time-limit truncation is reported separately and must not be counted as a fall.

Straight-line motion is expressed in the base's initial yaw frame. The report contains final and
maximum absolute lateral displacement, final and maximum absolute yaw drift, forward-velocity mean
absolute error, lateral-velocity mean absolute error, forward progress, fall rate, truncation rate,
and survival steps. Teacher-only diagnostics contain residual L2/L-infinity magnitude plus action
element and step clipping rates.

Every report retains nominal, left-knee, and right-knee strata. Metrics are measurements only until
the human accepts numeric thresholds; producing a report cannot by itself accept teacher quality.

## Required evidence

- Disabled actuator-strength configuration preserves the existing observation and SAC path.
- Enabled Phase-1 configuration changes only reset-time gains and appends exactly 29 privileged
  values to the critic observation, not the actor observation.
- Reset sampling covers nominal, left-knee, and right-knee cases with row-specific `g`.
- The nominal SAC checkpoint is frozen and bound into teacher checkpoint metadata by identity/hash.
- Deterministic zero residual equals nominal action; nonzero residual is bounded before final clip.
- Replay, learner, collector, and playback agree on the same privileged input and fused action.
- Paired evaluation proves exact branch-start identity and serializes per-stratum nominal, teacher,
  teacher-minus-nominal, and lower-is-better improvement measurements.

## Non-scope

6D Wrench teachers, Context Encoder architecture, rollout window definition, student distillation,
real-robot evaluation, publication novelty, and precise-trajectory acceptance remain unconfirmed.
