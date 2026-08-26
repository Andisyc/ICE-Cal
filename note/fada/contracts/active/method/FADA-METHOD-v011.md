---
contract_id: FADA-METHOD-v011
status: active
effective_date: 2026-08-26
supersedes: FADA-METHOD-v010
scope: unified-Oracle alternating Planner-IDM training
---

# FADA Planner-IDM Method Contract v011

The deployable interface remains `66/29/3`, `H=30`, `K=6`. One unified distillation checkpoint
(`dagger_iteration_8.pt` in the approved server campaign) is the only final Oracle authority for
walking, standing, and transition labels. Scenario-specific environments remain separate data
owners; they no longer imply separate policy authorities.

Each outer iteration executes two ordered optimizer units:

1. update IDM from trajectory and Oracle-shadow rows;
2. update Planner through the same IDM computation while IDM parameters are temporarily frozen.

The two optimizers never step together, but both units belong to the same campaign. Iteration zero
rolls out the unified Oracle; later iterations roll out the current Planner-IDM policy. The 20
intermediate SAC checkpoints provide IDM trajectory diversity only and never become Planner-label
authorities.

Source schema 4, scenario quotas `50/25/25`, cold-start quotas `50/50`, and 1:2 source retention
remain active. Training checkpoints use schema 5 and bind the schedule identity plus both optimizer
states. Schemas 1-4 remain inference-readable but cannot resume v011 training. Resume and warm start
remain disabled.

Offline tests do not claim convergence, stable locomotion, simulator quality, or authorization for
training, simulation, server mutation, or deployment.
