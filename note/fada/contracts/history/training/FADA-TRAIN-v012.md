---
contract_id: FADA-TRAIN-v012
status: historical
effective_date: 2026-08-26
supersedes: FADA-TRAIN-v011
method_contract: FADA-METHOD-v012
scope: serial privileged-Oracle lineage then fresh single-task Planner-IDM training
---

# FADA Source Training Contract v012

## Unit A — ICE-Cal privileged Oracle lineage

The official Oracle algorithm is generic `privileged_locomotion_sac`. The Actor directly consumes
the ordinary 98-D locomotion input plus a versioned privileged bundle containing base linear
velocity, ordered contact evidence, terrain/root-clearance evidence, ordered actuator state, and the
actual sampled domain-randomization parameters. The Critic receives at least the same information.

The Oracle owner seals the resolved observation layout, units/scales, body and joint order, task,
Reward, domain-randomization, G1 asset, MuJoCo backend, action scale, seed, and run lineage before
training. All sampled privileged values are cached numeric hot-path state. Asset/XML inspection is
cold-path only.

The first campaign uses 5,000 SAC iterations. It saves the twenty intermediate checkpoints at
`240, 480, ..., 4800` and the final Oracle at `5000`. Every artifact records the same
`oracle_lineage_id` and canonical configuration hashes. Checkpoint loading rejects a missing field,
layout drift, incompatible dimensions, wrong task/backend/action scale, or mixed lineage before
environment mutation.

The single task is `G1WalkFlat/MuJoCo`. Its owner configuration enables the FADA locomotion domain
randomization family. It sets `reward.scales.feet_phase=0` and rejects every non-zero phase-
conditioned gait-reward alias before environment creation.

Unit A ends after a separately authorized training campaign and policy-quality audit admit one final
Oracle plus exactly twenty source-compatible intermediate checkpoints. Completion of code or an
offline test cannot satisfy this gate.

## Unit B — fresh Planner–IDM source campaign

Unit B has one locomotion environment and one command distribution. It disables and rejects the
v011 standing environment, standing/transition curriculum, 50/25/25 scenario allocation, and
distillation-checkpoint final-Oracle loader. Optimal/suboptimal retention remains 1:2.

The source collector materializes `(X^H, A^H, Y^K, U^K, c)` with action-free 66-D state/future
fields. It stores final-Oracle shadows and the exact Oracle lineage. Replay admission rejects future
action leakage, source-role drift, mixed task/config/lineage identity, missing shadow labels, or a
count other than twenty unique intermediates.

Each iteration collects the current policy route, updates IDM, then updates Planner through a frozen
IDM. Iteration zero uses the final Oracle rollout; later iterations use the current Planner–IDM.
Intermediate checkpoints are collected as IDM-only trajectory sources throughout the same campaign.

v012 writes a new checkpoint schema binding the 66/95/29/3 observation contract, action-free future
identity, `alternating_idm_then_planner` schedule, Oracle lineage, task/Reward/DR hashes, counters,
metrics, and both optimizer states. Resume and warm start remain disabled for the first campaign.
Historical schema-5 checkpoints are inference-only and cannot initialize v012.

## Authority and evidence boundary

Oracle training, Planner–IDM training, simulator execution, server operation, Git publication, and
policy-quality evaluation each require their own explicit authority. The two training units are
serial; Unit B cannot launch from a merely present or shape-compatible Unit A checkpoint.
