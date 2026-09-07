---
contract_id: FADA-TRAIN-v023
status: historical / not-run
date: 2026-09-04
method_contract: FADA-METHOD-v023
superseded_by: FADA-TRAIN-v024
---

# FADA Training Contract v023

> Historical boundary (2026-09-07): this route was never run and cannot authorize new training.
> `mujoco_fada_phase` is retained only for compatibility and v024 baseline comparison.

## Stage A: phase-locomotion Oracle

Compose `task=sac/g1_walk_flat/mujoco_fada_phase`.
Set a new `ICE_CAL_ORACLE_LINEAGE_ID`; do not resume from a v022 lineage. The run must save the exact
20 checkpoints at iterations 240..4800 and `model_5000.pt`. All 21 records must use Oracle schema 3
and `behavior_profile=phase_locomotion_v1`.

## Stage B: walk-only Planner-IDM

Compose `task=g1_walk_flat/mujoco_fada_phase_locomotion`. Supply the final and intermediate v023
Oracle checkpoint paths from the same lineage. The output checkpoint is
`logs/fada/planner_idm_phase_locomotion_v023.pt` and must persist
`runtime_config.source_behavior_profile=phase_locomotion_v1`.

## Stage C/D: incline target and LoRA

The slope collection, adaptation, and evaluation roots select the phase-enabled target task and use
separate `*_phase_v023` artifact namespaces. Preflight rejects legacy phase-neutral or unknown source
behavior before environment construction or optimizer updates.

## Stop conditions

- Reject schema/profile-mixed Oracle lineages.
- Reject Stage B/C/D checkpoint behavior mismatch even when tensor dimensions agree.
- Do not substitute static-standing or walk-to-stand samples into this route.
- Do not claim policy quality from offline tests or configuration composition.
