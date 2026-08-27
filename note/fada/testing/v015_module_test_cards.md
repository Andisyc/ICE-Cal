# FADA v015 Module Test Cards

Status: human-confirmed on 2026-08-27; executable evidence pending.

## MTC-A — phase-neutral compatibility slots

- **Owner:** `G1WalkEnvCfg` and G1 walk environment phase lifecycle.
- **Public input/output:** resolved `gait_phase_enabled=false` task config → reset, observation, and
  action-step phase rows.
- **Ordinary case:** reset emits two zeros; observation exposes those zeros; stepping preserves them.
- **Boundary/identity:** batch and partial-reset rows retain shape `(N,2)` and Actor remains 98-D.
- **Invalid/sensitivity:** v014 `offset_phase` sampling and advancement must fail the zero assertions.
- **Profile:** S1/C1-C4 with T-value, T-shape, T-role, T-lifecycle, and T-diff.

## MTC-B — nominal dual-Reward profile

- **Owner:** Hydra task profile `mujoco_no_gait_dual_reward`.
- **Public input/output:** composed task config → standard SAC, nominal DR, phase-neutral 98-D task.
- **Ordinary case:** zero Command selects stand terms; nonzero Command selects walk terms.
- **Invalid:** any phase Reward, gait constraint, physical DR, privileged observation, or privileged
  runtime in the nominal profile is rejected by assertions.
- **Preference:** under zero Command, stable still support ranks above otherwise equal stepping;
  under nonzero Command, matched velocity ranks above otherwise equal stationary behavior.
- **Profile:** S1/C1-C4 with T-preference, T-role, T-config, and T-diff.

## MTC-C — final Oracle inheritance and downstream compatibility

- **Owner:** privileged Oracle Hydra profile and preflight.
- **Public input/output:** composed privileged profile → v015 admission or fail-closed `ValueError`.
- **Invariant:** it inherits the nominal phase-neutral task, then adds only privileged runtime and the
  existing left-knee Gain distribution; 98→66/29/3 stays unchanged.
- **Sensitivity:** overriding `env.gait_phase_enabled=true` must fail before environment creation.
- **Profile:** S1/C1-C4 with T-config, T-role, T-shape, and T-diff.
