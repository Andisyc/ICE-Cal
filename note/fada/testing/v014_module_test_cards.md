# FADA v014 Module Test Cards

Status: human-confirmed; executable evidence pending.

## MTC-A — gain-targeted source distribution

- **Owner:** Oracle Hydra profile consumed by `FADAPrivilegedSACRuntime.validate_training_config`.
- **Public input/output:** resolved training config → admission or fail-closed `ValueError` before
  environment creation.
- **Ordinary case:** only action index `3` is sampled, non-nominal `g in [0.8, 1.0]`, with nominal
  probability `0.3`.
- **Boundary cases:** `g=0.8`, `g=1.0`, and the exact nominal atom are valid.
- **Invalid cases:** any unrelated DR enabled, nonzero torque RFI, another actuator index, another
  range/probability, fixed sampling, duplicate explicit strength tail, or observation-noise drift.
- **Semantic properties:** S1/C1-C4 with T-value, T-role, T-dist, and T-diff.
- **Sensitivity:** the unchanged v013 full-DR profile must fail the new positive assertions, and
  each one-field override must fail preflight.

## MTC-B — privileged/deployable information boundary

- **Owner:** existing G1 reset actuator-strength path and typed FADA privileged observation.
- **Public input/output:** reset plan → Kp/Kd scale in the existing privileged Critic tail; Actor
  observation remains 98-D.
- **Invariant:** `include_in_critic_obs=false`; gain is not appended to Actor or duplicated as a
  second Critic tail.
- **Evidence:** existing actuator-strength reset tests plus Oracle profile/config tests.

## MTC-C — preserved Reward and Planner–IDM contract

- **Owner:** existing v013 no-gait dual-Reward preflight and v013 Planner–IDM boundaries.
- **Invariant:** command-owned stand/walk Reward, 20+1 lineage, 66/29/3 split, action-free future,
  serial IDM-before-Planner training, and first-action receding horizon are unchanged.
- **Evidence:** existing focused regressions; policy quality remains unclaimed.
