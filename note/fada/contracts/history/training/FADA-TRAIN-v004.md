---
contract_id: FADA-TRAIN-v004
status: superseded
effective_date: 2026-08-05
updated_date: 2026-08-05
supersedes: FADA-TRAIN-v003
method_contract: FADA-METHOD-v004
scope: persistent-async UniLab Planner-IDM DAgger with standing and walk-to-stand source curriculum
---

Superseded by `FADA-TRAIN-v005`, which closes the v004 cold-start coverage, Planner replay dilution,
and aggregate-quality blind spots found during closed-loop playback.

# FADA UniLab Source Training Contract

## Formal owner and isolation

`scripts/train_distill.py` owns the default-off route through `training.fada.enabled`.
`training.fada.stand_transition_curriculum.enabled` is the single owner flag for the new source
curriculum. With it disabled, collection remains the v003 route. With it enabled, the complete
scenario allocation, command schedule, standing Oracle requirement, metadata, and validation become
active together.

The ON route fails before environment creation unless the final walking Oracle, standing Oracle,
exactly 20 unique intermediate walking Oracle checkpoints, Oracle-shadow snapshot support, and the
2:1 intermediate-to-optimal budget are present. No walking-Oracle fallback is allowed for standing
or post-switch labels.

Formal execution remains `training.fada.execution_mode=persistent_async`. One spawned UniLab
collector owns resident `G1WalkFlat` and `G1StandStill` environments, resident walking and standing
final Oracles, transient intermediate walking Oracles, and rollout-side Planner-IDM. The parent owns
replay, optimizers, and persistence.

## Scenario allocation

`windows_per_iteration` remains the total optimal/current-policy budget. The curriculum owner divides
it deterministically across `walk`, `static_stand`, and `walk_to_stand`; configured ratios must be
finite, non-negative, sum to one, and every positive-ratio scenario must receive at least one window.
Rounding uses largest remainder with stable scenario order, so the total is exact.

`walk_to_stand` uses an active 3-D walking command for at least `H` steps, switches atomically to the
zero command, and admits only anchors whose history contains active-command rows and whose complete
future command is zero. The schedule cycles only after the configured post-switch horizon. Done rows
restart in the walking phase. `static_stand` forces zero command throughout collection.
`static_stand` must use the configured `standing_task` owner (currently
`g1_stand_still/mujoco` -> `G1StandStill`) so its reset velocity and initial-state distribution come
from standing training. `walk` and `walk_to_stand` remain in `G1WalkFlat`; a zero command in the
walking environment is not accepted as a replacement for the static-standing owner.

## Iteration barrier and source collection

Each outer iteration publishes one paired Planner-IDM state through `SharedWeightSync`. The worker
observes that version, collects all three optimal/current-policy scenarios plus the unchanged
intermediate walking sources, and writes one schema-validated artifact before the parent mutates
replay or either model.

Iteration zero executes the authoritative Oracle for each scenario. Later iterations execute the
current Planner-IDM while the same scenario Oracle supplies labels and same-state shadow pairs.
Intermediate walking Oracle rollouts remain a combined two times the total optimal/current-policy
budget and never replace standing Oracle Planner labels.

## IDM and Planner passes

IDM Eq. 4.2 averages first-action errors over valid realized trajectory and Oracle-shadow rows.
Planner Eq. 4.3 uses the scenario-authoritative Oracle first shadow action through the fixed IDM.
Scenario labels and quota counts are persisted for audit but do not become deployable model inputs.

## Quality, resume, and persistence

The v003 quality metrics and strict iteration barrier remain required. Source artifacts additionally
persist curriculum identity, exact per-scenario counts, Oracle role, and command-transition rejection
counts. Checkpoints persist the resolved curriculum configuration in `runtime_config`.

The v003 checkpoint may initialize a new v004 campaign only as model weights through an explicit
future initialization contract; it cannot resume optimizer/replay state. Current paper-exact
`resume_path` remains rejected because replay is not persisted.

## Non-scope

Training or modifying either Oracle, standing-to-walking transitions, target-domain collection,
LoRA, target adaptation, deployment, and later FADA stages remain outside this contract.
