# Distillation and G1 Hotspot Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the remaining distillation and G1 walk hotspots into focused production owners while preserving all runtime contracts.

**Architecture:** Keep current modules as compatibility/composition surfaces. Move cohesive implementation groups into acyclic sibling modules, retain one mutable owner for workflow and environment state, and use direct re-exports instead of duplicate implementations.

**Tech Stack:** Python 3.10, Hydra/OmegaConf, NumPy, PyTorch, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-30-distill-g1-hotspot-split-design.md`

## Global Constraints

- Use `uv run` for Python commands.
- Preserve the dirty working tree and `planner_idm_privileged_v022.pt`.
- Do not change training, tensor, environment, reward, action, DR, checkpoint, reset, Hydra, or playback semantics.
- Do not run simulation/training or perform branch, commit, push, deployment, remote, or destructive operations.
- Production owners must not import scripts or their compatibility facades; G1 siblings must not import `joystick`.

---

### Task 1: Freeze new owner boundaries

**Files:**
- Create: `tests/algos/test_distill_owner_boundaries.py`
- Create: `tests/envs/locomotion/g1/test_walk_owner_boundaries.py`

**Interfaces:**
- Consumes: existing public dataset, collector, workflow, and G1 environment behavior.
- Produces: executable compatibility and owner-location requirements for all extracted modules.

- [ ] Add import and real-value characterization tests for new owners and legacy aliases.
- [ ] Run both tests and confirm RED because the owner modules do not exist.

### Task 2: Split dataset and collector owners

**Files:**
- Create: `src/unilab/algos/torch/distill/dataset_contract.py`
- Create: `src/unilab/algos/torch/distill/dataset.py`
- Create: `src/unilab/algos/torch/distill/dataset_diagnostics.py`
- Create: `src/unilab/algos/torch/distill/dataset_merge.py`
- Create: `src/unilab/algos/torch/distill/dataset_io.py`
- Create: `src/unilab/algos/torch/distill/collection_common.py`
- Create: `src/unilab/algos/torch/distill/collection_standard.py`
- Create: `src/unilab/algos/torch/distill/collection_transition.py`
- Modify: `src/unilab/algos/torch/distill/data.py`
- Modify: `src/unilab/algos/torch/distill/collector.py`

**Interfaces:**
- Produces: unchanged dataset and collector call signatures through compatibility facades.
- Consumes: `DistillationBatch`, environment public reset/step state, and policy modules.

- [ ] Move dataset validation/value/diagnostics/merge/IO bodies without semantic edits.
- [ ] Move common, standard, and transition collection bodies without semantic edits.
- [ ] Run owner tests plus dataset, collection, persistent differential, and input-contract suites.

### Task 3: Split workflow and entry orchestration owners

**Files:**
- Create: `src/unilab/algos/torch/distill/workflow_contracts.py`
- Create: `src/unilab/algos/torch/distill/workflow_artifacts.py`
- Create: `src/unilab/algos/torch/distill/workflow_bootstrap.py`
- Create: `src/unilab/algos/torch/distill/workflow_dagger.py`
- Create: `src/unilab/algos/torch/distill/entry_plan.py`
- Create: `src/unilab/algos/torch/distill/workflow_diagnostics.py`
- Modify: `src/unilab/algos/torch/distill/workflow.py`
- Modify: `src/unilab/algos/torch/distill/entry_workflow.py`

**Interfaces:**
- Produces: unchanged workflow value types and public functions, plus a resolved internal entry plan.
- Consumes: dataset, collection, trainer, artifact paths, callbacks, and persistent collector lifecycle.

- [ ] Move contracts, persistence, bootstrap, and DAgger bodies to unique owners.
- [ ] Extract Hydra translation and diagnostic callbacks from the entry composition root.
- [ ] Run workflow, persistent worker, performance, and script route suites.

### Task 4: Split G1 walk decision owners

**Files:**
- Create: `src/unilab/envs/locomotion/g1/walk_observation.py`
- Create: `src/unilab/envs/locomotion/g1/walk_reward.py`
- Create: `src/unilab/envs/locomotion/g1/walk_commands.py`
- Create: `src/unilab/envs/locomotion/g1/walk_control.py`
- Modify: `src/unilab/envs/locomotion/g1/joystick.py`

**Interfaces:**
- Produces: unchanged `G1WalkEnv` framework hooks and explicit sibling decision owners.
- Consumes: validated config/context/arrays materialized by the environment owner.

- [ ] Move pure or explicit-input observation, reward, command, and control decisions.
- [ ] Keep backend access, reset snapshots, mutable rollout state, and framework hooks in `G1WalkEnv`.
- [ ] Run G1 observation, privileged, reward, gait, action-fault, DR, height, reset, registry, and backend tests.

### Task 5: Verify and close

**Files:**
- Create: `docs/superpowers/reviews/2026-08-30-distill-g1-hotspot-split-final-review.json`
- Modify: `docs/superpowers/execution/2026-08-30-distill-g1-hotspot-split-unit.json`

**Interfaces:**
- Consumes: complete authorized diff and fresh offline evidence.
- Produces: validated final-gate and completed one-shot receipts.

- [ ] Run all focused suites, Ruff, import-cycle probes, `git diff --check`, and compile checks.
- [ ] Inspect public aliases, state ownership, tensor provenance, persistence, and dependency direction.
- [ ] Recount physical lines and longest definitions; treat counts only as hotspot evidence.
- [ ] Validate the final review and execution unit manifests.
