# FADA v016 Module Test Cards

Status: human-confirmed by the v016 design confirmation and 2026-08-27 implementation authority.

## MTC-A — nominal single-Reward profile

- **Responsibility:** the nominal Hydra task owns one phase-neutral locomotion Reward for all commands.
- **Public boundary:** composed `mujoco_no_gait_single_reward` config.
- **Independent answer:** no runtime implementation or privilege; no enabled `reward.mode`; no key or
  active term beginning `stand_`; phase scales zero; gait constraint disabled.
- **Cases:** ordinary composed profile; zero phase boundary; forbidden nested mode/stand override;
  nominal-versus-privileged Reward identity.
- **Sensitivity:** the current v015 dual-Reward profile violates the absence assertions.
- **Forbidden shortcut:** do not infer correctness from total return or copy a second Reward formula.
- **Profile:** S0/S1, C1-C4, T-value/T-role/T-diff/T-preference.

## MTC-B — privileged single-Reward admission

- **Responsibility:** FADA privileged preflight admits only the inherited single Reward plus privilege/Gain.
- **Public boundary:** `validate_fada_single_reward` and
  `FADAPrivilegedSACRuntime.validate_training_config`.
- **Independent answer:** valid single Reward passes; enabled mode, nested `stand_*`, nonzero phase
  Reward, or gait constraint fails before environment creation.
- **Cases:** valid mapping, absent mode, malformed/nested forbidden authority, Hydra override.
- **Sensitivity:** the retired validator requires mode dispatch and therefore reverses v016 admission.
- **Forbidden shortcut:** no environment creation, simulator, private helper oracle, or silent fallback.
- **Profile:** S1, C1-C4, T-value/T-role/T-transform/T-diff.

## MTC-C — compatibility and lineage preservation

- **Responsibility:** existing G1 phase lifecycle, FADA input contract, and checkpoint gateway preserve
  v016's unchanged identities.
- **Public boundary:** phase-disabled reset/observation/step, 98→66/29/3 split, and 20+1 lineage API.
- **Independent answer:** two phase slots remain zero; Actor stays 98-D; split remains 66/29/3; final
  lineage remains checkpoints 240…4800 plus 5000.
- **Cases:** existing ordinary, boundary, identity and persistence regressions.
- **Sensitivity:** enabled gait clock, shifted tensor layout, or missing intermediate must fail.
- **Forbidden shortcut:** no checkpoint migration or compatibility padding.
- **Profile:** S1/S3, C1-C5, T-shape/T-order/T-role/T-persist/T-diff.
