# Stage 1 Optimization Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `code-construction` with TDD. This plan is
> executed inline because the user requested the diagnostic entrypoint in the current session.

**Goal:** Add a non-publishing Stage 1 entrypoint that distinguishes insufficient optimization from
an unrepresentable single-direction correction by recording learning progress on the sealed dataset.

**Architecture:** `calibration_training/stage1.py` remains the optimization owner. Production Stage 1
and diagnostics share one private optimizer-step function, while diagnostics own a temporary
Direction Bank and return immutable typed points. A thin script loads the existing checkpoint,
dataset, catalog, and identity, then prints one JSON report without writing a stage artifact.

**Tech Stack:** Python, PyTorch, argparse, dataclasses, pytest, Ruff, mypy, `uv run`.

---

## Boundary

- Preserve the active Stage 1 loss, frozen Planner/Tracker, split identities, source-projection
  validation, `0.1` publication gate, and artifact schema.
- Record checkpoints `0/25/100/300/1000` by default, with explicit CLI overrides allowed.
- Record axis index/name, step, training loss, training compensation ratio, validation compensation
  ratio, and raw Direction norm.
- Reject empty, negative, duplicate, or unordered checkpoint steps and non-finite/non-positive
  learning rates before optimizer construction.
- Do not normalize or publish the temporary Direction Bank. Normalization is composition-invariant;
  the raw norm is retained specifically as a diagnostic.
- Do not commit, push, run server training, or claim calibration quality in this unit.

## Task 1: Freeze the diagnostic contract with RED tests

**Files:**

- Modify: `tests/algos/test_fada_calibration_training.py`
- Modify: `tests/scripts/test_fada_calibration_entrypoints.py`

- [ ] Add a real-owner test using `_admitted_batch(policy)` and checkpoints `(0, 1)`. Assert six
  ordered points for three axes, finite metrics, step-zero Direction norm `0`, positive step-one
  norms, and bitwise-unchanged policy state, gradients, `requires_grad`, and evaluation mode.
- [ ] Add a Pinch Point characterization test that spies on `_direction_stage_step` and proves both
  `run_direction_stage_training` and `diagnose_direction_stage_training` call that exact owner. Use
  deterministic Direction values to prove checkpoint `n` metrics are read only after update `n`,
  from one consistent Direction Bank snapshot rather than an off-by-one or mixed snapshot.
- [ ] Inject an exception after a diagnostic update and prove policy parameters, gradients,
  `requires_grad`, and evaluation mode are restored before the exception escapes.
- [ ] Assert invalid checkpoint tuples fail before `torch.optim.Adam` construction.
- [ ] Poison `_atomic_torch_save` and prove diagnostics never call it. Make a diagnostic ratio exceed
  `0.1` and prove the point is still reported, while the existing formal Stage 1 gate still rejects
  the same ratio and publishes nothing.
- [ ] Add `diagnose_fada_calibration_stage1.py` to the CLI help inventory and require
  `--source-checkpoint`, `--dataset`, `--axis-catalog`, `--checkpoint-step`, and `--learning-rate`,
  while forbidding artifact and active-axis arguments.
- [ ] Run only these tests and observe RED because the diagnostic config, owner, and script do not
  exist.

## Task 2: Implement the owner-level diagnostic

**Files:**

- Modify: `src/unilab/algos/torch/fada_context/calibration_training/types.py`
- Modify: `src/unilab/algos/torch/fada_context/calibration_training/stage1.py`
- Modify: `src/unilab/algos/torch/fada_context/calibration_training/__init__.py`
- Modify: `src/unilab/algos/torch/fada_context/__init__.py`

- [ ] Add immutable `DirectionDiagnosticConfig` and `DirectionDiagnosticPoint` dataclasses. The
  config validates checkpoint ordering, learning rate, and split separation.
- [ ] Extract the existing single optimizer update into `_direction_stage_step`; keep the production
  loop behavior identical and preserve the existing frozen-owner and non-selected-axis checks.
- [ ] Add `diagnose_direction_stage_training(...) -> tuple[DirectionDiagnosticPoint, ...]`. Reuse
  `_split_stage_batch`, the exact optimizer step, `direction_stage_loss`, and
  `direction_stage_compensation_ratio`; record points before step zero and after each requested step.
- [ ] Keep the Direction Bank local, restore the borrowed policy on failure, and publish no file.
- [ ] Export only the typed config, point, and diagnostic owner through the existing public surfaces.
- [ ] Run the RED tests GREEN, followed by the complete calibration-training module tests.

## Task 3: Add the thin server entrypoint

**Files:**

- Create: `scripts/diagnose_fada_calibration_stage1.py`
- Modify: `tests/scripts/test_fada_calibration_entrypoints.py`

- [ ] Load the checkpoint on CPU, load the sealed dataset and catalog, verify source SHA, construct
  `CalibrationStageIdentity`, and call the diagnostic owner.
- [ ] Print one deterministic JSON object containing source/dataset/split identity, Axis Spec,
  learning rate, checkpoint steps, and typed diagnostic points with axis names.
- [ ] Do not accept an output artifact path and do not catch the owner's fail-closed exceptions.
- [ ] Add a `main()`-level test with real temporary checkpoint/dataset bytes and captured owner
  arguments. Prove source SHA, dataset SHA, split SHA, and exact Axis Spec reach the owner and the
  printed JSON unchanged. Prove a source SHA mismatch rejects before the diagnostic owner and
  `torch.optim.Adam` are reachable.
- [ ] Run CLI help and focused script tests GREEN.

## Task 4: Verify and review

- [ ] Run the impacted FADA calibration tests, Ruff, mypy, and `git diff --check`.
- [ ] Perform an R2 `code-review-expert` final gate over the exact diff, preserving the distinction
  between diagnostic connectivity and actual Stage 1 quality.
- [ ] Report the server command, but do not execute it or synchronize code.
