# Distillation Entrypoint and G1 Owner Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the distillation CLI and G1 walk hotspot by stable production responsibility while preserving every runtime and configuration contract.

**Architecture:** Keep scripts and registry modules as composition roots. Move cohesive implementation groups into acyclic production modules and preserve public imports through direct re-exports rather than duplicate wrappers.

**Tech Stack:** Python 3.10, Hydra/OmegaConf, NumPy, PyTorch, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-30-distill-g1-owner-split-design.md`

## Global Constraints

- Use `uv run` for all Python commands.
- Preserve the current dirty working tree and `planner_idm_privileged_v022.pt`.
- Do not change training, observation, reward, DR, checkpoint, reset, or playback semantics.
- Do not run simulation/training or perform Git writes.
- Production modules must not import scripts; G1 sibling modules must not import `joystick`.

---

### Task 1: Freeze compatibility and owner tests

**Files:**
- Modify: `tests/scripts/test_train_scripts.py`
- Modify: `tests/envs/test_env_configs.py`

**Interfaces:**
- Consumes: current `scripts.train_distill` callables and G1 joystick exports.
- Produces: tests requiring the new production owners and unchanged public aliases.

- [ ] Add assertions that public script callables resolve to production-owner functions after extraction.
- [ ] Add assertions that G1 configuration, math, and DR symbols remain importable from `joystick` and are owned by their sibling modules.
- [ ] Run the new tests and observe failure because the owner modules do not exist.

### Task 2: Extract distillation training and collection routes

**Files:**
- Create: `src/unilab/algos/torch/distill/entry_training.py`
- Create: `src/unilab/algos/torch/distill/entry_collection.py`
- Modify: `scripts/train_distill.py`
- Modify: affected script tests to patch the production owner for internal seams.

**Interfaces:**
- Produces: existing `build_teacher_spec`, `build_student_policy`, trainer/update functions, `run_collect_dataset`, and `run_online_dagger_update` signatures unchanged.
- Consumes: existing distill data, trainer, backend, and policy APIs.

- [ ] Move model/runtime/update definitions into `entry_training.py` without changing bodies.
- [ ] Move collection contracts and routes into `entry_collection.py` without changing bodies.
- [ ] Import those callables into the script composition root.
- [ ] Run focused distillation script collection and update tests.

### Task 3: Extract the single-entry workflow

**Files:**
- Create: `src/unilab/algos/torch/distill/entry_workflow.py`
- Create: `src/unilab/algos/torch/distill/workflow_transition.py`
- Modify: `scripts/train_distill.py`
- Modify: `tests/scripts/test_train_scripts.py`
- Modify: `tests/scripts/test_stand_height_walk_distill_workflow.py`

**Interfaces:**
- Produces: `run_single_entry_workflow(cfg, *, persistent_scenario_collector_factory=None, performance_clock=time.perf_counter)`.
- Consumes: training and collection routes plus existing workflow lifecycle owners.

- [ ] Move role/spec/config assembly and workflow orchestration to `entry_workflow.py`.
- [ ] Move legacy walk-to-stop environment collection to `workflow_transition.py`.
- [ ] Keep lifecycle cleanup, resume/fork, logger, checkpoint, and performance behavior identical.
- [ ] Patch internal tests at the new owner and run both workflow suites.

### Task 4: Extract G1 configuration, math, and DR owner

**Files:**
- Create: `src/unilab/envs/locomotion/g1/walk_config.py`
- Create: `src/unilab/envs/locomotion/g1/walk_math.py`
- Create: `src/unilab/envs/locomotion/g1/walk_domain_randomization.py`
- Modify: `src/unilab/envs/locomotion/g1/joystick.py`
- Modify: owner-location tests only.

**Interfaces:**
- Produces: existing G1 config classes, deterministic calculation functions, and `G1WalkDomainRandomizationProvider` with unchanged signatures.
- Consumes: base environment/backend contracts and common locomotion DR APIs.

- [ ] Move pure deterministic functions to `walk_math.py`.
- [ ] Move configuration dataclasses to `walk_config.py` without changing defaults.
- [ ] Move the DR provider to `walk_domain_randomization.py` without changing state or reset behavior.
- [ ] Re-export all legacy names from `joystick.py` and keep registry identities unchanged.
- [ ] Run G1 config, actuator-strength, gait-constraint, height, action-fault, and backend DR tests.

### Task 5: Verify and review the complete diff

**Files:**
- Create: `docs/superpowers/reviews/2026-08-30-distill-g1-owner-split-final-review.json`
- Modify: `docs/superpowers/execution/2026-08-30-distill-g1-owner-split-unit.json`

**Interfaces:**
- Consumes: complete source/test diff and fresh command output.
- Produces: validated final-gate and completed one-shot receipts.

- [ ] Run focused pytest suites for every moved owner.
- [ ] Run Ruff on all changed Python files and `git diff --check`.
- [ ] Import every new module and verify the declared dependency graph is acyclic.
- [ ] Recount physical lines and longest definitions.
- [ ] Inspect the complete diff for semantic changes, duplicate owners, and unrelated edits.
- [ ] Validate the final review and execution manifests.
