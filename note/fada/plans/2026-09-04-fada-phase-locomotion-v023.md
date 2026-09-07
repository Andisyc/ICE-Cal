# FADA v023 Phase-Locomotion Source Plan

Status: historical / superseded by FADA-METHOD-v024 on 2026-09-07. This plan was not run and cannot
authorize training. `mujoco_fada_phase` remains only a compatibility and comparison profile.

## Accepted behavior

Replace the active FADA source-policy route with a dedicated locomotion-only privileged Oracle
whose deployable observation contains an advancing gait phase. The Oracle is not responsible for
static standing or walk-to-stand behavior. Preserve the v022 route and checkpoints as an explicit
phase-neutral legacy baseline.

The new route adopts FADA's locomotion observation contract from Table 5 and the existing UniLab
G1WalkFlat gait clock and `feet_phase=5.0` reward. It does not add a straight-line corridor reward,
change the Planner-IDM architecture, change the H=30/K=6 first-action objective, or alter target
LoRA parameter ownership.

## Engineering boundary

1. Add one versioned Oracle behavior profile, `phase_locomotion_v1`, next to the legacy
   `phase_neutral_mixed_v1` profile. The profile owner validates phase, command, and reward
   semantics before environment creation.
2. Seal the profile in Oracle checkpoints. New schema-3 Oracle checkpoints require it; schema-2
   checkpoints have the sole explicit interpretation `phase_neutral_mixed_v1`. Mixed-profile and
   mixed-schema lineages fail closed.
3. Add a v023 Oracle Hydra owner by inheriting the sealed grouped-DR lineage and overriding only
   phase/command/reward semantics. Grouped DR, privileged normalization, iteration schedule, and
   20+1 persistence remain unchanged.
4. Add a v023 Planner-IDM Hydra owner. It uses the v023 Oracle, enables gait phase, sets standing
   and transition ratios to zero, disables the stand-transition curriculum and v005 mixed-scenario
   replay, and writes to a new artifact/checkpoint namespace.
5. Persist `source_behavior_profile` in schema-5 Planner-IDM runtime metadata. Stage C/D configs
   require `phase_locomotion_v1`; old checkpoints without the field are interpreted only as the
   phase-neutral legacy profile.
6. Add a phase-enabled Stage-C task owner and route slope collection, adaptation, and evaluation to
   separate v023 paths. Target randomization remains disabled and slope geometry remains unchanged.

## Proof route

- Hydra composition tests prove old-profile equivalence and the complete new profile.
- Semantic pseudo-samples prove gait-phase slots survive 98-to-66 projection and advance under the
  environment's existing gait-clock owner.
- Checkpoint and Collector matrices prove old/old and new/new admission plus old/new rejection.
- Stage-B tests prove the source allocation contains only `walk`.
- Stage-C/D preflights prove the checkpoint profile and environment profile match before rollout or
  optimizer mutation.
- Focused FADA tests, impacted regressions, Ruff, Pyright on changed files, and diff checks close the
  offline engineering unit. No training, simulation, policy-quality, deployment, commit, or push is
  authorized by this plan.
