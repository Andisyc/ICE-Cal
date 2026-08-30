# Final Three-Hotspot Architecture Implementation Plan

> **For agentic workers:** Execute inline in one authorized transaction. Steps
> use checkbox syntax for evidence tracking; no branch, commit, live run, or
> external operation is authorized.

**Goal:** Remove the final three confirmed maintainability defects while
preserving all runtime-visible behavior.

**Architecture:** Direct owner dependencies replace internal aggregate imports;
standard collection becomes an explicit transaction; G1 framework hooks move
to stateless responsibility-specific binding owners while `G1WalkEnv` retains
the single mutable lifecycle.

**Tech Stack:** Python 3, PyTorch, NumPy, Hydra/OmegaConf, pytest, Ruff, `uv run`.

**Spec:** `docs/superpowers/specs/2026-08-31-final-three-hotspot-architecture-design.md`

## Global constraints

- Preserve training/tensor/gradient/optimizer/dataset/checkpoint/Hydra/reward/
  observation/action/reset/DR/RNG behavior and legacy public imports.
- Use `uv run`; preserve the dirty tree; no Git writes or live work.
- New modules must remove a named dependency or responsibility; no generic
  utility dumping and no thin compatibility layer with independent behavior.

### Task 1: Establish RED architecture fitness tests

- Modify `tests/algos/test_distill_package_consolidation.py` to parse production
  owner imports and reject the aggregate package edge.
- Modify `tests/algos/test_distill_owner_boundaries.py` to require
  `StandardCollectionTransaction` as the owner behind the unchanged public
  function.
- Modify `tests/envs/locomotion/g1/test_walk_owner_boundaries.py` to require
  observation/control/runtime binding owners in the `G1WalkEnv` MRO.
- Run the three files and confirm failures identify only the missing boundaries.

### Task 2: Correct Distill dependency direction

- Modify workflow owner imports in `workflows/entry_training.py`,
  `entry_collection.py`, `entry_workflow.py`, and `transition.py` to import
  concrete owners directly.
- Search all owner packages for the forbidden aggregate import.
- Run package, workflow, and script entrypoint tests.

### Task 3: Extract the standard collection transaction

- Create `collection/standard_transaction.py` containing the explicit request,
  mutable transaction state, validation, step, admission, and finalization
  phases migrated from the existing function.
- Modify `collection/standard.py` to retain the public signature and delegate to
  the transaction owner.
- Preserve private compatibility seams required by existing tests via direct
  owner aliases only.
- Run standard collection, dataset contract, async collection, and failure-path
  tests.

### Task 4: Extract G1 framework binding owners

- Create `walk_observation_bindings.py`, `walk_control_bindings.py`, and
  `walk_runtime_bindings.py` by moving the corresponding unchanged Env hooks.
- Modify `joystick.py` imports and MRO; leave configuration materialization,
  base initialization, registration, and the unique mutable Env state there.
- Keep all pure calculations in the existing `walk_observation.py`,
  `walk_control.py`, `walk_commands.py`, and other established owners.
- Run G1 owner, observation, privilege, action, curriculum, symmetry, reset,
  reward, and backend-contract tests.

### Task 5: Close the complete engineering unit

- Run scoped Ruff and compileall.
- Run architecture tests, all affected suites, import sweep, and repository
  pytest; classify only verified pre-existing out-of-scope failures.
- Recount major files/functions and confirm the changes reduced responsibility
  and dependency knowledge rather than merely redistributing lines.
- Validate the R2 final review and one-shot completion receipts.
