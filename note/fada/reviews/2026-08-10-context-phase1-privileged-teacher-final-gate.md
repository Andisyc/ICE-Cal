# Context Phase-1 Privileged Teacher Final Gate

Date: 2026-08-10
Mode: `migration_review`
Verdict: engineering closure accepted; formal teacher-quality training not yet accepted

## Findings

No P0-P2 implementation blocker remains in the bounded Phase-1 route.

P3: the SAC entropy term describes the stochastic residual policy while the environment receives the
clipped sum of nominal and residual actions. This approximation is acceptable for the low-alpha
feasibility run, but saturation frequency should be measured before formal training; excessive
clipping would make residual-policy entropy a poor description of executed-action diversity.

P3: `g` currently scales Kp and Kd together. Results must be described as gain-based actuator
effectiveness repair and must not be generalized to torque saturation, delay, backlash, thermal
derating, or a measured real motor fault without a stronger actuator model.

## Isolation review

- One explicit route owner: `algo.runtime_impl=privileged_residual_sac`.
- Standard SAC/HORA and 98/101D G1 paths remain default-off and regression-tested.
- The nominal actor is frozen, eval-only, hash-bound, and tensor-checked whenever actor state loads.
- The residual actor is the only actor optimizer owner.
- Actor observation excludes `g`; replayed critic observation carries the final 29D `g`.
- Fixed left-knee `0.9` remains a playback-only profile and is not used as the teacher training
  distribution.

## Next gate

Before a formal teacher run, define paired same-seed trajectory metrics for nominal versus teacher,
including lateral displacement, yaw drift, forward-command tracking error, fall rate, and residual
clipping rate. Survival alone cannot accept the teacher. Only after teacher improvement is measured
should the project define rollout contents and begin Context Encoder distillation.
