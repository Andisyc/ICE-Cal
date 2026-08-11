---
contract_id: FADA-CONTEXT-PHASE1-METHOD-v003
status: superseded
effective_date: 2026-08-10
updated_date: 2026-08-10
supersedes: FADA-CONTEXT-PHASE1-METHOD-v002
superseded_by: FADA-CONTEXT-PHASE1-METHOD-v004
prerequisite: FADA-METHOD-v005
scope: minimal privileged residual-teacher feasibility under nominal and fixed left-knee-0.9 conditions
---

# FADA Context Phase-1 Method Contract

This contract governs the first feasibility experiment for execution repair. It does not claim that
a rollout-conditioned Context Encoder is trained, identifiable, or deployment-ready.

## Teacher boundary

The simulator owns a true 29D actuator-strength vector `g`. The vector changes reset-time position
servo gains and is also exposed to the privileged teacher. The frozen walking SAC checkpoint owns
the nominal action. The teacher may output only a bounded additive residual:

```text
nominal_action = frozen_walking_sac(actor_observation)
delta_action = privileged_residual_teacher(actor_observation, g)
executed_action = clip(nominal_action + delta_action, -1, 1)
```

The teacher must never replace or update the nominal walking policy. Zero deterministic residual must
reproduce the nominal actor exactly. Training, collection, evaluation, and playback must use the same
actor-owned fusion rule.

Phase-1 training uses exactly two structural strata with equal reset probability: nominal strength
and exactly one changed left-knee actuator at multiplier `0.9`. Right-knee faults and continuous
strength generalization are outside this feasibility stage. Training only on the anomalous stratum
would allow a future Context Encoder to collapse to a constant output, so nominal rows remain
mandatory even though the target anomaly itself is fixed.

The future Context Encoder does not receive `g` and does not regress `g`. Its future output is a
temporary latent execution Context inferred from deployable rollout history. Distillation of that
latent path is outside this contract.

## Paired measurement owner

Teacher feasibility is measured against the frozen nominal actor from the exact same initialized
environment snapshot. Both branches receive the same `g`, command, physics state, observation,
history carrier, and RNG state. Autoreset is disabled during each branch. A physical termination is
a fall; a time-limit truncation is reported separately and must not be counted as a fall.

Straight-line motion is expressed in the base's initial yaw frame. The report contains final and
maximum absolute lateral displacement, final and maximum absolute yaw drift, forward-velocity mean
absolute error, lateral-velocity mean absolute error, forward progress, fall rate, truncation rate,
and survival steps. Teacher-only diagnostics contain residual L2/L-infinity magnitude plus action
element and step clipping rates.

Every report retains nominal and fixed-left-knee-0.9 strata. The accepted formal protocol uses
held-out seeds `[101, 102, 103, 104, 105]`, `256` environments per seed, `400` steps, and command
`(0.4, 0.0, 0.0)` m/s. Each per-stratum metric is first summarized within a seed and then averaged
equally across the five seeds. Overall measurements remain diagnostic and cannot accept the teacher.

## Accepted quality gate

All checks below are conjunctive. A positive reduction means lower teacher error.

For `left_knee`:

- maximum absolute lateral displacement reduction is at least `10%`;
- maximum absolute yaw-drift reduction is at least `10%`;
- forward-velocity MAE is no more than `2%` worse than nominal;
- teacher fall rate is no greater than nominal and no greater than `1%` absolute;
- teacher action clipping step rate is no greater than `1%`.

For the nominal stratum, teacher final/maximum lateral error, final/maximum yaw error, forward-
velocity MAE, and lateral-velocity MAE are each no more than `2%` worse than nominal. Nominal-stratum
teacher fall rate is no greater than nominal and no greater than `1%`; clipping step rate is no
greater than `1%`.

Non-finite metrics, missing strata, inexact pairing, or a protocol-shape mismatch produce
`unassessed` or `failed`, never `passed`. A one-update or shortened sentinel cannot establish quality.

## Required evidence

- Disabled actuator-strength configuration preserves the existing observation and SAC path.
- Enabled Phase-1 configuration changes only reset-time gains and appends exactly 29 privileged
  values to the critic observation, not the actor observation.
- Reset sampling covers nominal and fixed-left-knee-0.9 cases with row-specific `g`.
- The nominal SAC checkpoint is frozen and bound into teacher checkpoint metadata by identity/hash.
- Deterministic zero residual equals nominal action; nonzero residual is bounded before final clip.
- Replay, learner, collector, and playback agree on the same privileged input and fused action.
- Paired evaluation proves exact branch-start identity and serializes per-stratum nominal, teacher,
  teacher-minus-nominal, lower-is-better improvement, and machine-readable gate results.
- Formal training may start only after the v003 training preflight passes without creating a trainer
  process, run directory, or checkpoint.

## Non-scope

Right-knee faults, continuous strength generalization, 6D Wrench teachers, Context Encoder
architecture, rollout window definition, student distillation, real-robot evaluation, publication
novelty, and formal teacher quality remain unconfirmed until the trained checkpoint passes this gate.
