---
contract_id: FADA-METHOD-v023
status: active
date: 2026-09-04
supersedes_for_slope_demo: FADA-METHOD-v022
paper: FADA Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control
---

# FADA Method Contract v023

## Objective

Train a dedicated privileged G1 locomotion Oracle for the FADA straight-line incline experiment.
Unlike the v022 baseline, this policy does not own static standing or walk-to-stand behavior.

## Source behavior

- Behavior identity is `phase_locomotion_v1`.
- The task remains `G1WalkFlat` on MuJoCo with one locomotion Reward.
- Actor input remains the 98-D task observation plus normalized privileged information; Critic input
  remains the exact typed privileged observation.
- Gait Phase is active, initialized with `offset_phase`, and advances at 1.5 Hz.
- The phase reward uses the mature UniLab G1 settings: `feet_phase=5.0`, swing height `0.09`, and
  tracking sigma `0.04`. Phase contrast/contact terms and the gait constraint remain disabled.
- `commands.rel_standing_envs=0.0`; transition samples and command resampling remain disabled.
- The v022 grouped dynamics-randomization schedule and 20 intermediate plus one final Oracle
  lineage are retained.
- No additional straight-line, lateral-position, or yaw Reward is introduced.

## Observation and distillation

The raw Actor observation stays 98-D. Its final two values are `sin(Phi), cos(Phi)`, and the existing
`g1_fada_state_v2` projection preserves them in the 66-D student observation. Planner-IDM therefore
keeps the existing architecture, H=30, K=6, and action-free future contract.

Stage B is walk-only: `stand_transition_curriculum.enabled=false` and `v005_replay.enabled=false`.
All source windows come from the same phase-locomotion Oracle lineage.

## Artifact compatibility

Oracle checkpoint schema 3 records `behavior_profile`. Legacy schema 2 is interpreted only as
`phase_neutral_mixed_v1`. A 20+1 lineage may not mix schemas or behavior profiles. Planner-IDM
schema-5 checkpoints persist `source_behavior_profile`; Stage C/D reject a same-shape checkpoint
whose profile does not match their configured route.

FADA-METHOD-v022 remains the named phase-neutral mixed-behavior baseline and is not rewritten.

## Evidence boundary

Offline composition and unit tests can establish contract wiring only. Training stability, gait
quality, straight-line tracking, slope adaptation, and policy quality require separately authorized
runtime experiments.
