# Playback and Distillation Transaction Owner Split Implementation Plan

> **For agentic workers:** execute inline under code-construction with strict RED/GREEN checkpoints.

**Goal:** Complete one behavior-preserving owner split for playback and the remaining FADA
transaction hotspots.

**Architecture:** Keep public entrypoints and mutable lifecycle owners stable. Extract only
cohesive overlay/control/trace decisions and transaction phases with explicit inputs and outputs.

**Tech Stack:** Python, PyTorch, NumPy, Hydra/OmegaConf, MuJoCo, pytest, Ruff, `uv run`.

**Spec:** `docs/superpowers/specs/2026-08-30-playback-distill-transaction-owner-split-design.md`

## Global constraints

- Preserve all current user changes in the dirty main checkout.
- Use `uv run`; do not start playback, simulation, collection, or training.
- Do not branch, commit, push, publish, delete artifacts, or change schemas/config values.
- Keep scripts as composition roots and backend-specific rendering under visualization.

## Task 1: Freeze owner boundaries

- Add characterization tests importing the planned playback owner modules and exercising real
  trace formatting, command construction, and overlay math.
- Add transaction characterization tests for public result identity, row order, and commit-last
  behavior using existing fake environments and workflow fixtures.
- Run the tests before production modules exist and record the expected RED result.

## Task 2: Extract playback overlay, controls, and trace owners

- Create `src/unilab/visualization/playback_overlay.py`.
- Create `src/unilab/visualization/playback_controls.py`.
- Create `src/unilab/visualization/playback_trace.py`.
- Modify `scripts/play_interactive.py` to import/re-export those helpers and retain CLI/viewer
  composition plus compatibility monkeypatch seams.
- Run visualization and interactive-script tests.

## Task 3: Reduce interactive playback factory coupling

- Extract stable playback contracts/routing only where caller injection and import compatibility
  remain exact.
- Keep `interactive_playback.py` as the compatibility facade.
- Run `tests/visualization/test_interactive_playback.py`, FADA playback tests, training helper
  tests, and visualization entrypoint tests.

## Task 4: Split collection transactions

- Refactor transition collection into explicit prepare, step/append, and finalize phases with one
  mutable transaction owner.
- Refactor FADA source collection into spec/reset preparation, per-step Oracle/rollout/window work,
  and result finalization without changing history or compaction order.
- Run transition, persistent differential, FADA source collection, replay/admission, persistence,
  unified Oracle, and refactor-boundary tests.

## Task 5: Split merge and DAgger iteration phases

- Divide source parsing/validation, optional-field consistency, and final dataset construction in
  `dataset_merge.py` without adding another module.
- Extract one typed DAgger iteration phase while preserving atomic commit as the final durable
  action.
- Run G1 distillation contract and DAgger workflow tests.

## Task 6: Final evidence and review

- Run the complete affected playback/distillation/G1 suite.
- Run Ruff, compileall, `git diff --check`, forbidden reverse-import searches, and AST line recount.
- Validate one `code-review-expert/v1` final-gate receipt and one completed
  `one-shot-execution/v1` receipt.
- Report residual hotspots without claiming live runtime, training, or policy quality.
