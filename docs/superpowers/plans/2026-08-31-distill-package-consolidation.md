# Distillation Package Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated local hotspot splits with one behavior-preserving package consolidation across distillation, playback, and the G1 environment boundary.

**Architecture:** Existing implementations move into responsibility-based production subpackages. Current repository import paths remain direct re-export facades only where they have consumers; orchestration, state, persistence, diagnostics, and framework bindings keep distinct owners.

**Tech Stack:** Python 3.10, PyTorch, Hydra, NumPy, MuJoCo adapters, pytest, Ruff, `uv run`.

**Spec:** `docs/superpowers/specs/2026-08-31-distill-package-consolidation-design.md`

## Global Constraints

- Preserve the dirty working tree based at commit `41d37e957906fdbaaa73bca5ef1f77f3aa2efe73`.
- Do not change model, tensor, gradient, optimizer, dataset, collection, checkpoint, Hydra, reward, observation, action, reset, DR, RNG, simulator, or playback behavior.
- Do not create a branch, commit, push, train, simulate, deploy, or perform destructive cleanup.
- Always use `uv run` for Python commands.
- Keep scripts as composition roots and backend-specific behavior behind existing backend contracts.
- Compatibility modules contain direct imports/re-exports only; no fallback or duplicated implementation.
- Internal task checkpoints are not user approval gates; repair in-scope offline failures within this transaction.

---

### Task 1: Freeze executable compatibility and owner boundaries

**Files:**
- Create: `tests/algos/test_distill_package_consolidation.py`
- Create: `tests/visualization/test_playback_package_consolidation.py`
- Create: `tests/envs/locomotion/g1/test_walk_reward_bindings.py`
- Inspect: `src/unilab/algos/torch/distill/__init__.py`
- Inspect: `scripts/train_distill.py`
- Inspect: `scripts/play_interactive.py`

**Interfaces:**
- Consumes: current public symbols and behavior from the approved dirty-tree baseline.
- Produces: failing import-identity tests for the new production owner paths and characterization coverage for framework-bound G1 rewards.

- [ ] Record current public module/symbol importers and the internal dependency graph.
- [ ] Add literal import-identity cases such as legacy `distill.trainer.BehaviorDistillationTrainer` being the same object as `distill.learning.trainer.BehaviorDistillationTrainer`.
- [ ] Add representative dataset, collection, workflow, runtime, and FADA owner-path cases; each must fail because the target package does not exist.
- [ ] Add playback factory identity cases against `visualization.playback_sessions` and viewer owner paths.
- [ ] Add reward binding behavior cases using existing reward fixtures, not source-text assertions or mocks of the method under test.
- [ ] Run the three new test files with `uv run pytest ... -q` and retain the expected missing-owner RED failures.

### Task 2: Establish contracts and dataset production packages

**Files:**
- Create: `src/unilab/algos/torch/distill/contracts/__init__.py`
- Move implementation to: `contracts/checkpoint.py`, `contracts/dataset.py`, `contracts/workflow.py`
- Create: `src/unilab/algos/torch/distill/datasets/__init__.py`
- Move implementation to: `datasets/dataset.py`, `datasets/diagnostics.py`, `datasets/io.py`, `datasets/merge.py`
- Modify or retain as direct facades: `checkpoint.py`, `dataset.py`, `dataset_contract.py`, `dataset_diagnostics.py`, `dataset_io.py`, `dataset_merge.py`, `data.py`
- Modify: internal imports and `distill/__init__.py`

**Interfaces:**
- Consumes: existing dataset/checkpoint/workflow types and functions without signature changes.
- Produces: `distill.contracts.*` and `distill.datasets.*` owner paths plus compatible legacy imports.

- [ ] Move the existing bodies without editing validation, values, metadata, ordering, or IO behavior.
- [ ] Rewrite production dependencies toward `contracts` and `datasets`; neither package imports workflow, learning, runtime, playback, or scripts.
- [ ] Replace used legacy modules with explicit re-exports and remove unused internal-only legacy modules.
- [ ] Run package RED tests until dataset/contract cases turn GREEN.
- [ ] Run `uv run pytest tests/algos/test_distill_dataset.py tests/algos/test_distill_dataset_contract.py tests/algos/test_distill_dataset_merge.py -q` using the actual existing test filenames selected by `rg --files tests/algos | rg 'distill.*dataset|dataset.*distill'`.

### Task 3: Consolidate collection and runtime lifecycle

**Files:**
- Create: `src/unilab/algos/torch/distill/collection/__init__.py`
- Move implementation to: `collection/common.py`, `collection/standard.py`, `collection/transition.py`, `collection/transition_state.py`
- Create: `src/unilab/algos/torch/distill/runtime/__init__.py`
- Move implementation to: `runtime/async_runtime.py`, `runtime/persistent_resources.py`, `runtime/persistent_runtime.py`, `runtime/g1_worker.py`
- Modify or retain as direct facades: corresponding legacy root modules and `collector.py`
- Modify: internal collection/runtime imports.

**Interfaces:**
- Consumes: contracts and datasets from Task 2.
- Produces: typed collection result and runtime lifecycle owners without trainer or workflow mutation.

- [ ] Move standard and transition collection implementations while preserving pre-step labels, row order, resets, counters, compaction, and finalization.
- [ ] Move persistent process/resource ownership while preserving environment and policy lifetime, exception transport, and teardown.
- [ ] Remove imports from collection into workflow entry modules or runtime implementations.
- [ ] Convert legacy modules with consumers to explicit re-export facades.
- [ ] Run collection/runtime import tests and all tests selected by `rg --files tests | rg 'distill.*collect|collect.*distill|persistent.*distill|async.*distill'`.

### Task 4: Consolidate learning and workflow transactions

**Files:**
- Create: `src/unilab/algos/torch/distill/learning/__init__.py`
- Move implementation to: `learning/models.py`, `learning/teacher.py`, `learning/trainer.py`, `learning/routing.py`, `learning/diagnostics.py`, `learning/offline.py`, `learning/dagger.py`, `learning/moe_student.py`, `learning/moe_diagnostics.py`
- Create: `src/unilab/algos/torch/distill/workflows/__init__.py`
- Move implementation to: `workflows/entry_plan.py`, `workflows/entry_training.py`, `workflows/entry_collection.py`, `workflows/entry_workflow.py`, `workflows/artifacts.py`, `workflows/bootstrap.py`, `workflows/dagger.py`, `workflows/dagger_iteration.py`, `workflows/diagnostics.py`, `workflows/runtime.py`, `workflows/transition.py`, `workflows/formal_identity.py`
- Create dependency-neutral owners: `observability/debug.py`, `observability/performance.py`; this implementation refinement breaks dataset/trainer/workflow import cycles without changing telemetry behavior.
- Modify or retain as direct facades: corresponding legacy root modules and `workflow.py`
- Modify: `scripts/train_distill.py` only if an owner import changes.

**Interfaces:**
- Consumes: contracts, datasets, collection, and runtime owners from Tasks 2–3.
- Produces: stable trainer/update and workflow entry surfaces with one optimizer owner and commit-last persistence.

- [ ] Move trainer and offline bodies unchanged; keep models, optimizers, gradient execution, update count, and teacher evaluation in `BehaviorDistillationTrainer`.
- [ ] Move workflow composition and durable artifact phases without moving validation or persistence into entry scripts.
- [ ] Update internal consumers to production paths and public consumers to stable legacy or package exports.
- [ ] Ensure diagnostics remain read-only and no collection owner imports workflow state.
- [ ] Run trainer/offline/workflow import tests and all tests selected by `rg --files tests | rg 'distill.*(trainer|offline|workflow|dagger)|stand_height_walk_distill'`.

### Task 5: Consolidate the complete FADA family

**Files:**
- Create: `src/unilab/algos/torch/distill/fada/__init__.py`
- Move `fada.py` implementation to: `fada/model.py`
- Move every `fada_*.py` production implementation under `distill/fada/` using concise names that preserve the existing suffix identity, including adaptation, async collection/runtime/config, checkpoint, collection contract/IO/state/transaction/windows, legacy/persistent workflow, observation, Oracle, playback, privileged Oracle/SAC, replay, source artifact/diagnostics/evaluation/plan, target collector/data, trainer/training/diagnostics, windows, workflow/setup.
- Retain explicit legacy `fada_*.py` facades only for paths with repository consumers.
- Modify: `distill/__init__.py` and all internal FADA imports.

**Interfaces:**
- Consumes: generic stable contract, dataset, collection, learning, runtime, and workflow interfaces.
- Produces: `distill.fada` package exports and compatible legacy FADA module paths.

- [ ] Replace the former flat `fada.py` with the package owner while preserving every symbol exported by `distill.__init__`.
- [ ] Move FADA files by responsibility without changing Oracle lineage, checkpoint schema, observation contract, source/target rows, adaptation, replay, or diagnostic calculations.
- [ ] Rewrite FADA-to-generic dependencies toward the new package owners and prevent generic packages from importing FADA workflows.
- [ ] Convert externally consumed legacy paths to explicit re-export facades; remove internal-only old paths.
- [ ] Run package identity tests and all tests selected by `rg --files tests | rg 'fada|distill'`.

### Task 6: Finish playback composition

**Files:**
- Modify: `scripts/play_interactive.py`
- Modify: `src/unilab/visualization/interactive_playback.py`
- Modify: `src/unilab/visualization/playback_sessions.py`
- Modify: `src/unilab/visualization/playback_distill_policy.py`
- Modify: `src/unilab/visualization/playback_distill_routing.py`
- Create: `src/unilab/visualization/playback_checkpoint_contract.py`
- Create: `src/unilab/visualization/playback_cli.py`
- Create: `src/unilab/visualization/playback_viewer.py`
- Preserve: `playback_controls.py`, `playback_overlay.py`, `playback_trace.py`

**Interfaces:**
- Consumes: existing playback session factories, policy/checkpoint loaders, environment factory, viewer adapters, controls, overlays, and trace values.
- Produces: a CLI/viewer composition script and compatibility session facade without duplicated factory bodies.

- [ ] Move checkpoint/run-config contract interpretation out of the script into `playback_checkpoint_contract.py`.
- [ ] Move CLI dataclasses, parsing, override normalization, and config composition into `playback_cli.py`.
- [ ] Move viewer resource preparation, frame rendering, loop, and guaranteed cleanup into `playback_viewer.py`.
- [ ] Move algorithm-specific factories out of `interactive_playback.py` into `playback_sessions.py` and distillation routing modules.
- [ ] Keep old callable names as direct imports so script and test consumers remain compatible.
- [ ] Run playback RED/GREEN tests and all tests selected by `rg --files tests | rg 'play.*interactive|interactive.*play|playback'`.

### Task 7: Finish the G1 framework binding boundary

**Files:**
- Create: `src/unilab/envs/locomotion/g1/walk_reward_bindings.py`
- Modify: `src/unilab/envs/locomotion/g1/joystick.py`
- Inspect without semantic edits: existing `walk_reward.py`, `walk_observation.py`, `walk_control.py`, `action_trace.py`
- Test: `tests/envs/locomotion/g1/test_walk_reward_bindings.py`

**Interfaces:**
- Consumes: existing pure reward functions and the reward registry requirement for bound environment methods.
- Produces: one stateless `G1WalkRewardBindings` framework adapter inherited by `G1WalkEnv`; mutable environment state remains owned by `G1WalkEnv`.

- [ ] Move only reward registry binding/context/dispatch methods into `G1WalkRewardBindings`; move no mutable fields or lifecycle transitions.
- [ ] Keep calculations in `walk_reward.py` and pass explicit configuration/state projections.
- [ ] Keep backend handles, rollout state, reset, commands, curriculum, observation hook, and action hook on `G1WalkEnv`.
- [ ] Run the reward binding tests and all G1 environment/reward/observation/action/DR tests selected by `rg --files tests/envs | rg 'g1|walk|joystick'`.

### Task 8: Remove transitional duplication and close the architecture

**Files:**
- Modify: all compatibility `__init__.py` and root facades created or retained above.
- Modify: package consumers under `src/`, `scripts/`, and `tests/` where private implementation imports must use production owner paths.
- Create: `docs/superpowers/reviews/2026-08-31-distill-package-consolidation-final-review.json`
- Create: `docs/superpowers/execution/2026-08-31-distill-package-consolidation-unit.json`

**Interfaces:**
- Consumes: completed owner packages and the frozen compatibility list.
- Produces: an acyclic production dependency graph, one implementation per symbol, validated offline closeout receipts, and no in-scope follow-up split.

- [ ] Import every production and compatibility module in a clean subprocess and fail on cycles or missing exports.
- [ ] Verify legacy and production imports return identical objects for all frozen public symbols.
- [ ] Scan root facades for assignments, function/class definitions, error handling, fallback, mutable state, and durable IO; remove any remaining business implementation.
- [ ] Recount production owner files/functions and inspect every threshold exception for cohesive responsibility rather than mechanically splitting it.
- [ ] Run focused package suites, all directly affected tests, and then the full repository test command used by the current checkout.
- [ ] Run Ruff, `uv run python -m compileall -q src scripts tests`, dependency-cycle checks, and `git diff --check`.
- [ ] Compare failures with the frozen baseline; introduce no new failure or deselection.
- [ ] Validate the final `code-review-expert` R2 receipt and one-shot execution unit.
- [ ] Report the complete changed boundary and evidence without claiming simulation, training, checkpoint quality, or policy quality.
