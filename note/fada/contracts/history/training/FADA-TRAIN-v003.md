---
contract_id: FADA-TRAIN-v003
status: superseded
effective_date: 2026-08-05
updated_date: 2026-08-05
supersedes: FADA-TRAIN-v002
method_contract: FADA-METHOD-v003
scope: paper-aligned UniLab source-domain Planner-IDM training through iterative DAgger
---

# FADA UniLab Source Training Contract

Superseded by `FADA-TRAIN-v004`, which adds a default-off static-standing and walk-to-stand
curriculum with explicit standing Oracle authority. The text below preserves the v003 contract.

## Formal owner and isolation

`scripts/train_distill.py` owns the default-off route through `training.fada.enabled`. When the
paper-exact source flag is active, source training fails before environment creation unless the final
Oracle, exactly 20 unique intermediate Oracle checkpoints, Oracle-shadow snapshot support, and a
2:1 suboptimal-to-optimal budget are all present.

Formal execution uses `training.fada.execution_mode=persistent_async`. One spawned UniLab collector
process owns the environment, final Oracle, transient intermediate Oracle, and rollout-side
Planner-IDM. The parent process owns replay, optimizers, checkpoint persistence, and the ordered IDM
then Planner update. The prior synchronous formal run is terminated and is not an accepted result.

## Iteration barrier

Each outer iteration publishes one paired Planner-IDM state through `SharedWeightSync`. The
persistent collector must observe that exact version before it collects the optimal/current-policy
source and all 20 intermediate-Oracle sources into one schema-validated artifact. Only after the
artifact returns may the parent mutate replay or update either model.

## Source collection

Each visited state stores deployable history, complete command, realized rollout fields, the complete
same-state final-Oracle shadow pair, and its first action label. Shadow rollout executes inside a
transaction that restores simulator physics, sensors, environment state/info, counters, RNG, pending
forces, and autoreset state even on failure. Shadow rows crossing termination or command changes are
invalid and may not enter the Oracle-source IDM loss.

Iteration zero uses final-Oracle rollout; later iterations use Planner-IDM first actions. In every
iteration, intermediate Oracle rollouts contribute a combined window count equal to twice the
optimal/current-policy window count, distributed across 20 checkpoint identities.

## IDM and Planner passes

IDM Eq. 4.2 averages first-action errors over the union of valid realized trajectory rows and valid
Oracle-shadow rows. Planner Eq. 4.3 uses each visited state's final-Oracle first shadow action through
the fixed IDM. Intermediate rollout actions never replace final-Oracle Planner labels.

## Quality evidence

Every saved training checkpoint records finite source-quality metrics for realized future to executed
action through IDM, Oracle-shadow future to final-Oracle action through IDM, Planner future through
IDM to final-Oracle action, Planner future deviation from realized future, and rollout rejection
counts. These metrics prove boundary quality only; stability requires separate closed-loop evidence.

## Resume and persistence

The prior v002 checkpoint is evidence only and cannot resume a v003 paper-exact campaign. Until the
bounded replay itself is checkpoint-persisted, v003 paper-exact training rejects non-null
`resume_path`. Collector artifacts are atomically persisted and validated again in the parent.

## Non-scope

Target-domain collection, LoRA, target adaptation, deployment, and later FADA stages remain outside
this contract.
