# Distill and G1 Deep Owner Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the remaining confirmed distillation and G1 walk maintainability hotspots while preserving every accepted runtime and research-ML behavior.

**Architecture:** Stateful lifecycle owners remain in place. Pure decisions and transaction phases move behind direct, typed module functions; compatibility import and method seams remain thin delegations.

**Tech Stack:** Python 3, NumPy, PyTorch, Hydra-owned configuration, pytest, Ruff, compileall.

**Spec:** `docs/superpowers/specs/2026-08-30-distill-g1-deep-owner-split-design.md`

## Global Constraints

- Always run repository Python commands with `uv run`.
- Preserve all current dirty and untracked user work.
- Do not create a branch, commit, push, train, simulate, deploy, or perform external writes.
- Do not change model, reward, observation, action, DR, RNG, reset, dataset, checkpoint, or Hydra semantics.
- Do not add mixins, generic managers, service frameworks, or duplicated mutable owners.

---

### Task 1: Freeze owner boundaries with RED tests

**Files:**
- Create: `tests/algos/test_distill_deep_owner_boundaries.py`
- Create: `tests/envs/locomotion/g1/test_walk_deep_owner_boundaries.py`

**Interfaces:**
- Consumes: current trainer, dataset merge, collection, workflow, reward, and DR behavior.
- Produces: executable import and numerical contracts for the new pure owners.

- [ ] Add real behavior tests for routing, source compatibility, reward terms, actuator range validation/sampling, and reset decisions with hand-derived expected values.
- [ ] Add import-direction assertions for the new production owners and existing compatibility entrypoints.
- [ ] Run the two new files and observe failure caused only by missing new owner modules/functions.

### Task 2: Split Trainer routing and diagnostics

**Files:**
- Create: `src/unilab/algos/torch/distill/trainer_routing.py`
- Create: `src/unilab/algos/torch/distill/trainer_diagnostics.py`
- Modify: `src/unilab/algos/torch/distill/trainer.py`
- Test: `tests/algos/test_distill_deep_owner_boundaries.py`
- Test: existing distillation trainer tests.

**Interfaces:**
- Produces: pure target-index resolution and read-only diagnostic snapshot helpers.
- Preserves: `BehaviorDistillationTrainer` API and optimizer/gradient ownership.

- [ ] Move pure routing validation and index materialization without changing error conditions, device, dtype, or trace timing.
- [ ] Move read-only runtime snapshot/emission helpers without allowing diagnostics to affect training state.
- [ ] Keep teacher/student calls, loss assembly, backward, clipping, optimizer step, and update count in `trainer.py`.
- [ ] Run new owner tests and existing trainer/offline tests to GREEN.

### Task 3: Decompose long dataset, collection, and DAgger transactions

**Files:**
- Modify: `src/unilab/algos/torch/distill/dataset_merge.py`
- Modify: `src/unilab/algos/torch/distill/collection_transition.py`
- Modify: `src/unilab/algos/torch/distill/workflow_dagger.py`
- Test: `tests/algos/test_distill_deep_owner_boundaries.py`
- Test: existing dataset, collection, and workflow suites.

**Interfaces:**
- Produces: named validation, execution, finalization, and commit phases inside each established owner.
- Preserves: public call signatures, callback order, row order, failure behavior, resume identity, and artifact commit order.

- [ ] Extract validated-source loading and final concatenation phases from dataset merge.
- [ ] Extract transition row buffer/finalization phases while keeping environment stepping in the collector entry function.
- [ ] Extract one DAgger iteration and durable commit phases while keeping workflow resume and outer iteration ownership in `run_multirole_dagger_workflow`.
- [ ] Run direct transaction suites to GREEN after each extraction.

### Task 4: Move stateless G1 reward decisions

**Files:**
- Modify: `src/unilab/envs/locomotion/g1/walk_reward.py`
- Modify: `src/unilab/envs/locomotion/g1/joystick.py`
- Test: `tests/envs/locomotion/g1/test_walk_deep_owner_boundaries.py`
- Test: existing gait, height, actuator, and issue-175 regression suites.

**Interfaces:**
- Produces: stateless reward functions consuming `RewardContext`, explicit configuration, or explicit arrays.
- Preserves: environment reward registry and private compatibility methods as thin delegates.

- [ ] Classify reward methods by dependency and move only methods with no mutable environment/backend ownership.
- [ ] Keep context construction, reward dispatch, logging, curriculum, and backend-derived projections in `G1WalkEnv`.
- [ ] Run numerical owner tests and all direct reward suites to GREEN.

### Task 5: Split pure DR calculations

**Files:**
- Create: `src/unilab/envs/locomotion/g1/walk_actuator_randomization.py`
- Create: `src/unilab/envs/locomotion/g1/walk_reset_randomization.py`
- Modify: `src/unilab/envs/locomotion/g1/walk_domain_randomization.py`
- Test: `tests/envs/locomotion/g1/test_walk_deep_owner_boundaries.py`
- Test: existing actuator-strength, gait-constraint, and DR-provider suites.

**Interfaces:**
- Produces: pure actuator validation/range/sampling and reset mask/decision helpers.
- Preserves: one public provider, provider persistence, exact RNG call order, and reset payload structure.

- [ ] Move only calculations whose inputs and outputs can be explicit without copying provider state.
- [ ] Keep curriculum mutation, capture/restore, backend-neutral plan assembly, and public lifecycle in the provider.
- [ ] Run direct DR tests to GREEN.

### Task 6: Full offline verification and final review

**Files:**
- Create: `docs/superpowers/reviews/2026-08-30-distill-g1-deep-owner-split-diff-manifest.txt`
- Create: `docs/superpowers/reviews/2026-08-30-distill-g1-deep-owner-split-final-review.json`
- Update: `docs/superpowers/execution/2026-08-30-distill-g1-deep-owner-split-unit.json`

**Interfaces:**
- Produces: current test, static-boundary, line-count, and maintainability receipts.

- [ ] Run the new owner tests and the complete directly affected test set.
- [ ] Run Ruff, compileall, `git diff --check`, reverse-import checks, and AST line/function recount.
- [ ] Review the coherent diff under standard, module-boundary, repository-discipline, and research-ML profiles.
- [ ] Validate `FINAL_GATE_PASS` and the one-shot `complete` contract.

