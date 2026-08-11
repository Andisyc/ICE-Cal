# Planner-IDM Construction Plan

Status: completed on 2026-08-04

Terminal outcome: paper-aligned, reusable Planner and IDM modules with source-loss boundaries and focused deterministic tests.

Scope: `src/unilab/algos/torch/distill/fada.py`, public exports, focused tests, FADA contracts and Architecture projections.

Non-scope: rollout collector integration, simulator snapshot restore, task-specific command schedules, target adaptation, LoRA, deployment, evaluation, or long training.

## Step 1 / 1

Objective: implement the complete generic Planner-IDM architecture and Eq. 4.2/Eq. 4.3 owner boundaries.

Owner files/modules:

- `src/unilab/algos/torch/distill/fada.py`
- `src/unilab/algos/torch/distill/__init__.py`
- `tests/algos/test_fada_planner_idm.py`

Expected evidence: syntax/import check, six focused model/loss tests, Atlas contract check, coherent-diff review. Closed by `../evidence/2026-08-04-planner-idm-construction.md`.

Stop condition: all local deterministic checks pass, or the first environment/dependency blocker is recorded without claiming runtime completion.

Why one step: model construction, loss ownership, tests, and documentation share one reversible local authority boundary and one terminal outcome.
