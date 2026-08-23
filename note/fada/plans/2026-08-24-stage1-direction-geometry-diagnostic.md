# Stage 1 Direction Geometry Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `code-construction` with
> `superpowers:test-driven-development`. This plan is executed inline because the user explicitly
> approved the direction-collinearity diagnostic.

**Goal:** Determine whether gain rows admit one shared execution-relevant latent correction or
require conflicting per-row corrections.

**Architecture:** Add one diagnostic-only owner beside Stage 1. It reuses the sealed dataset,
source-projection checks, frozen policy, and the existing `first_action_mse` semantic owner. Because
the Tracker Decoder is a per-token linear `action_head`, it solves one shared and independent
per-row minimum-norm first-action directions analytically through the Decoder pseudoinverse. It then
reports exact fit ratios plus signed consensus cosine, direction-norm dispersion, and singular-value
evidence. A thin CLI prints JSON and has no artifact output path.

**Tech Stack:** Python, PyTorch linear algebra, argparse, dataclasses, pytest, Ruff, mypy, `uv run`.

---

## Boundary

- Preserve formal Stage 1/2/3 training, the current `0.1` gate, schemas, and publication behavior.
- Diagnose only the executed first action, using `first_action_mse` as the confirmed semantic owner.
- Exclude and count rows with zero coefficient or zero uncompensated first-action error because they
  do not identify a correction direction.
- Fit one shared direction on training rows and evaluate it on train/validation rows.
- Fit independent row directions only as diagnostic oracles; never publish or feed them into later
  stages.
- Report top-1 uncentered energy and signed cosine-to-consensus together, because singular energy
  alone cannot distinguish aligned from opposite directions.
- Do not repair the discovered full-chunk Stage 1/2 supervision mismatch in this unit.
- Do not commit, push, synchronize, or run the server command.

## Task 1: Freeze Geometry Semantics With RED Tests

**Files:**

- Modify: `tests/algos/test_fada_calibration_training.py`
- Modify: `tests/scripts/test_fada_calibration_entrypoints.py`

- [ ] Add independent hand-oracle cases for aligned, orthogonal, and opposite direction matrices.
  Require aligned directions to have top-1 energy and signed consensus cosine of one; require the
  orthogonal case to have top-1 energy one half; require the opposite case to retain top-1 energy one
  while exposing a nonzero opposing fraction.
- [ ] Add an owner-level diagnostic case with checkpoints reduced to one step. Assert exact axis and
  split identity, finite metrics, zero-coefficient accounting, and unchanged policy parameters,
  gradients, `requires_grad`, and module modes.
- [ ] Add failure cases for fewer than two identifiable directions and non-finite direction data.
- [ ] Add the new CLI to the parser inventory. Require checkpoint, dataset, and catalog arguments;
  forbid optimizer, output, artifact, and active-axis arguments.
- [ ] Add a `main()` provenance test proving checkpoint SHA, dataset SHA, split SHA, exact Axis Spec,
  and typed reports reach deterministic JSON. Prove source mismatch rejects before the diagnostic
  owner and optimizer.
- [ ] Run the focused tests and observe RED because the geometry types, owner, and CLI do not exist.

## Task 2: Implement the Diagnostic Owner

**Files:**

- Create: `src/unilab/algos/torch/fada_context/calibration_training/direction_geometry.py`
- Modify: `src/unilab/algos/torch/fada_context/calibration_training/types.py`
- Modify: `src/unilab/algos/torch/fada_context/calibration_training/__init__.py`
- Modify: `src/unilab/algos/torch/fada_context/__init__.py`

- [ ] Add immutable config and report dataclasses with split separation and a positive coefficient
  identifiability threshold.
- [ ] Reuse `_split_stage_batch` so batch schema, source projection, role selection, and exact Axis
  Spec width fail closed before optimization.
- [ ] Build frozen nominal latents once. Use the linear Decoder pseudoinverse to solve one
  training-shared token-zero direction and independent token-zero directions for each split under
  `first_action_mse` semantics. Prove the diagnostic constructs no optimizer.
- [ ] Compute per-row compensation ratios, gate fraction, direction norm, top-1 energy, signed
  cosine-to-consensus quantiles, and opposing-direction fraction from identifiable rows only.
- [ ] Keep every trainable diagnostic tensor local and verify the borrowed policy is unchanged.
- [ ] Run the RED owner tests GREEN.

## Task 3: Add the Thin JSON CLI

**Files:**

- Create: `scripts/diagnose_fada_calibration_direction_geometry.py`
- Modify: `tests/scripts/test_fada_calibration_entrypoints.py`

- [ ] Load the source checkpoint on CPU, dataset, and catalog; verify source digest before invoking
  the owner; bind dataset and split digests plus exact Axis Spec in `CalibrationStageIdentity`.
- [ ] Print deterministic JSON with `supervision_scope=executed_first_action`, optimizer settings,
  provenance, axis names, and typed reports.
- [ ] Accept no output path and call no persistence owner.
- [ ] Run focused CLI tests GREEN.

## Task 4: Verify and Review

- [ ] Run the changed test files and the impacted FADA calibration suite.
- [ ] Run Ruff, mypy, and `git diff --check`.
- [ ] Perform an R1 `code-review-expert` construction review over the exact diagnostic diff.
- [ ] Report the uploadable server command without executing or synchronizing it.
