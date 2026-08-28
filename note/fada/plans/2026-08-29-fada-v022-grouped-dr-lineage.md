# FADA v022 Grouped-DR Teacher Lineage Plan

> Status: CURRENT / IMPLEMENTATION PENDING

## Objective

Preserve the successful live-privileged grouped-DR teacher behavior while replacing its
non-persistent validation profile with one sealed source-lineage profile that writes all 20 IDM
coverage checkpoints and the final Oracle checkpoint.

## Preserved behavior

- `G1WalkFlat/MuJoCo`, one locomotion Reward, gait phase disabled.
- Live normalized `g1_fada_privileged_v1` input to the SAC Actor and Critic.
- 5000 iterations and the existing penalty curriculum.
- Group levels at `0, 500, 1200, 2000, 3000, 4000` with scales
  `0.0, 0.2, 0.4, 0.6, 0.8, 1.0`.
- Existing actuator-strength, Kp/Kd, friction, mass, COM, and DoF-bias ranges.
- Delay and pushes disabled.

## Required engineering change

1. Add one sealed grouped-DR lineage mode owned by the privileged-Oracle runtime/config.
2. Admit `save_interval=240` without weakening the grouped curriculum validation.
3. Persist checkpoints `240…4800` and `5000` through `FADAOracleCheckpointGateway` under one
   `oracle_lineage_id`.
4. Keep the current validation profile unchanged for diagnostics.
5. Prove exact checkpoint identity and rejection of missing/mixed iterations before server training.

## Current evidence and blocker

The v022 validation run showed high Reward and episode length in Electerm, but exact metrics were not
sealed into a repository artifact. The subsequent IDM launch failed because the 20 intermediate
checkpoint files did not exist. Current configuration explains this deterministically:
`privileged_dr_curriculum_validation=true` selects `checkpoint_mode=validation` and
`save_interval=1000`.

The first unclosed boundary is therefore checkpoint persistence, not privileged-input learning or
domain-randomization policy quality.

## Stop conditions

- Any change alters the successful curriculum levels, ranges, Reward, or privileged normalization.
- A command claims IDM readiness before all 21 checkpoints pass lineage admission.
- Historical v017 nominal-only semantics are reintroduced into the active v022 route.
- Training begins without a focused config/preflight test and formal route check.
