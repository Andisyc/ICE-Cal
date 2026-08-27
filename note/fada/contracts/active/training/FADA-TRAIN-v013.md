---
contract_id: FADA-TRAIN-v013
status: active
effective_date: 2026-08-27
supersedes: FADA-TRAIN-v012
method_contract: FADA-METHOD-v013
scope: serial privileged-Oracle lineage then fresh single-task Planner-IDM training
---

# FADA Source Training Contract v013

## Unit A — privileged Oracle

The official route remains `privileged_locomotion_sac` on one `G1WalkFlat/MuJoCo` environment.
The typed privileged observation, cached domain-randomization state, 5,000-iteration schedule, and
20+1 checkpoint lineage remain as defined by v012.

The owner config sets `mode_observation=false`, `rel_standing_envs=0.3`,
`rel_transition_envs=0`, and uses command-conditioned `RewardModeConfig` dispatch:

- common balance: orientation, angular-velocity, action-rate, foot-orientation, and alive terms;
- stand/static and stand/recovery: the same support/stability term set, with no action-magnitude or
  rigid stand-still term;
- walk: linear/angular command tracking and ordinary pose regularization;
- gait/phase scales zero and `gait_constraint.enabled=false`, `penalty_scale=0`.

Preflight validates branch membership, active nonzero scales, isolation of stand and walk terms,
absence of forbidden standing terms, and disabled gait mechanisms. Resolved Reward configuration is
sealed by the existing Oracle checkpoint hash.

## Unit B — Planner–IDM

Unit B retains the established source collection, tensor, optimizer, persistence, and admission
semantics: action-free future, final-Oracle label ownership, intermediate IDM-only coverage, and
IDM-before-Planner serial updates.
It remains blocked until a separately authorized formal runtime audit and policy-quality audit admit
the v013 final Oracle and its twenty same-lineage intermediate checkpoints.

## Authority

Training, simulation, server operation, Git publication, deployment, and policy-quality evaluation
remain separate explicit actions. Local module evidence cannot authorize a long run.
