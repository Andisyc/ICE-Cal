---
contract_id: FADA-CONTEXT-PHASE1-METHOD-v005
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FADA-CONTEXT-PHASE1-METHOD-v004
prerequisite: FADA-CONTEXT-METHOD-v001
scope: fixed-left-knee-0.9 privileged full-action teacher with forward-progress enforcement
superseded_by: FADA-CONTEXT-PHASE1-METHOD-v006
---

# FADA Context Phase-1 Full-Action Teacher v005 Contract

This contract retains the v004 full-action teacher and exact paired comparison, while rejecting the
stationary reward shortcut found by v004.

## Unchanged policy boundary

```text
baseline: actor_observation -> frozen original walking SAC -> complete 29D action
teacher:  actor_observation + true 29D motor-strength g -> trained teacher -> complete 29D action
```

Both policies execute under the same fixed left-knee index `3 = 0.9` physics. The teacher remains an
independent, fully trainable policy initialized from the original actor; it contains no nominal
branch and no residual action fusion.

## Forward-progress enforcement

The v004 teacher reduced lateral/yaw error by standing nearly still. v005 makes this shortcut an
episode failure during training:

- record reset-time base position and yaw;
- after `50` environment steps, compute average forward speed in the reset yaw frame;
- for commands with forward component at least `0.1 m/s`, terminate an episode when average forward
  speed is below `0.20 m/s`;
- keep existing tilt/height termination as a logical OR with this new condition;
- leave the termination disabled by default for every other G1 task.

The threshold is runtime-calibrated against the frozen original actor at fixed left-knee `0.9`.
Across held-out seeds `[101, 102, 103, 104, 105]`, `256` environments per seed, its minimum average
speed at step `50` was `0.226277 m/s`; the threshold therefore preserves about `11.6%` margin at the
earliest enforcement step. The rejected v004 teacher averaged about `-0.0032 m/s`.

## Quality gate

The v004 formal paired gate remains unchanged. On the same five held-out seeds, `256` environments,
`400` steps, command `(0.4, 0.0, 0.0)`, and fixed left-knee `0.9`, all checks are conjunctive:

- maximum lateral displacement reduction at least `10%`;
- maximum yaw-drift reduction at least `10%`;
- forward-velocity MAE no more than `2%` worse than baseline;
- teacher fall rate no greater than baseline and no greater than `1%`;
- teacher action saturation step rate no greater than `1%`.

The evaluator must use the v005 task owner so the same progress-failure semantics apply to both
branches. Training completion alone is not acceptance.

## Required evidence

- Disabled progress termination preserves all existing G1 termination behavior.
- Warm-up rows never terminate for progress; forward commands below `0.1 m/s` are excluded.
- At and after step `50`, average speed below `0.20 m/s` terminates while equal/greater speed passes.
- Reset yaw, not world `x`, defines forward displacement.
- The original actor survives a bounded fixed-`0.9` MuJoCo sentinel without progress termination.
- A stationary full-action policy is terminated at the declared boundary.
- Full-action actor, privilege, checkpoint, runner, and paired-evaluation contracts remain unchanged.

## Non-scope

Context Encoder training, latent repair, generalized motor strengths, right-leg faults, reward-scale
tuning, and publication claims remain outside this contract.
