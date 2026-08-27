---
contract_id: FADA-TRAIN-v017
status: active
effective_date: 2026-08-27
supersedes: FADA-TRAIN-v016
method_contract: FADA-METHOD-v017
scope: nominal privileged Oracle, Planner-IDM distillation, then downstream failed-rollout collection
---

# FADA Source Training Contract v017

## Unit A — nominal privileged Oracle

Train one privileged SAC Oracle on the single `G1WalkFlat/MuJoCo` task under strictly nominal
dynamics. The Actor remains deployable-observation-only; the Critic retains the typed privileged
state interface. No execution Gain, actuator attenuation, delay, bias, or other physical/domain
randomization is enabled. The existing successful standard-SAC base-chain evidence is a regression
reference, not a second Oracle or a separate checkpoint lineage.

Only this nominal privileged run produces checkpoints `240…4800` plus `5000` under one
`oracle_lineage_id`. Admission requires survival and command-tracking policy-quality evidence; a
less-negative return with short episodes or persistent 100% termination fails.

## Unit B — Planner–IDM

After Unit A passes formal-runtime and policy-quality gates, freeze its final policy and use the same
lineage for Planner–IDM collection. Intermediate checkpoints expand IDM suboptimal-policy coverage;
the final Oracle provides all source action labels. Preserve the 98→66/29/3 input contract,
action-free future, causal future–action pairing, IDM-before-Planner ordering, and frozen-IDM Planner
gradient path.

## Unit C — failed-rollout and calibration data

Only after the source Oracle and Planner–Tracker are frozen may collection inject left-knee Gain or
another declared execution fault. Perturbation parameters belong to the failed rollout and its
calibration evidence; they never enter the Oracle lineage.

## Current transition state

The repository still implements the superseded v016 gain-targeted Oracle configuration. Therefore
v017 is `TRANSITION-BLOCKED` before engineering: no formal audit, server run, long training,
checkpoint reuse, Planner–IDM training, or policy-quality claim is currently authorized.
