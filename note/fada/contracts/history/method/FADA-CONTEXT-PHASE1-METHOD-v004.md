---
contract_id: FADA-CONTEXT-PHASE1-METHOD-v004
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FADA-CONTEXT-PHASE1-METHOD-v003
superseded_by: FADA-CONTEXT-PHASE1-METHOD-v005
prerequisite: FADA-CONTEXT-METHOD-v001
scope: fixed-left-knee-0.9 privileged full-action teacher and paired nominal comparison
---

# FADA Context Phase-1 Full-Action Teacher Contract

This contract governs a deliberately narrow retry requested by the human. It tests whether a
privileged policy trained directly in the fixed left-knee `0.9` environment can produce a more
precise straight trajectory than the original walking policy executed in that same environment.

## Compared policies

Both branches use the same fixed physical condition, command, initial simulator state, environment
carrier, and RNG state:

```text
baseline: actor_observation -> frozen original walking SAC -> complete 29D action
teacher:  actor_observation + true 29D motor-strength g -> trained teacher -> complete 29D action
```

Left-knee actuator index `3` has multiplier `0.9`; every other multiplier is `1.0`. This condition
is applied to both branches. Passing `g` means giving it only to the teacher policy as privileged
input. The original policy never consumes `g`; it experiences the same `0.9` physics through the
environment.

The teacher is initialized from the original walking actor where tensor interfaces permit, but it
is an independent full-action policy and all of its policy parameters are trainable. There is no
frozen nominal actor inside the teacher forward pass and no action addition or residual fusion.

## Training scope

- Train only on the fixed left-knee `0.9` condition; no nominal rows are required for this teacher.
- Use the existing G1 walking observations plus the exact 29D `g` as teacher policy inputs.
- Use the existing UniLab synchronized double-buffer off-policy runner and SAC objective.
- Keep the fixed `(0.4, 0.0, 0.0)` m/s command and initial-yaw-frame straight-line rewards.
- Preserve the original walking checkpoint as the baseline identity and warm-start source.

This experiment does not test whether a future Context Encoder uses history. Constant `g` is
acceptable for this teacher-only retry; distinguishable rollout conditions remain mandatory later
for Context identifiability.

## Paired evaluation

The evaluator must snapshot one fixed-`0.9` environment state, run the original policy, restore the
exact snapshot, and run the full-action teacher. It reports final/maximum lateral displacement,
final/maximum yaw drift, forward/lateral velocity error, forward progress, falls, truncations,
survival, action difference, and action saturation.

Formal teacher quality requires all of the following on held-out seeds `[101, 102, 103, 104, 105]`,
`256` environments per seed, `400` steps, and command `(0.4, 0.0, 0.0)` m/s:

- teacher maximum lateral displacement is at least `10%` lower than baseline;
- teacher maximum yaw drift is at least `10%` lower than baseline;
- teacher forward-velocity MAE is no more than `2%` worse than baseline;
- teacher fall rate is no greater than baseline and no greater than `1%`;
- teacher action saturation step rate is no greater than `1%`.

Training completion or survival alone does not establish teacher quality.

## Required evidence

- The fixed strength vector is exactly `(1, 1, 1, 0.9, 1, ..., 1)` in physics and replay.
- Teacher actor input has dimensions `(actor_obs=98, g=29)` and outputs `(action=29)`.
- Teacher actor output contains no baseline-action addition in its runtime dataflow.
- Nominal warm start is numerically equal to the original actor before training for matched inputs.
- All teacher actor parameters receive the actor optimizer; the original checkpoint is never updated.
- Collector, learner, checkpoint loader, playback, and evaluation agree on the full-action contract.
- Same-snapshot evaluation proves exact branch-start equality and fixed-`0.9` condition equality.

## Non-scope

Context Encoder implementation, latent `delta_z` training, motor-strength generalization, nominal
teacher performance, right-leg anomalies, real-robot deployment, and publication claims are outside
this contract.
