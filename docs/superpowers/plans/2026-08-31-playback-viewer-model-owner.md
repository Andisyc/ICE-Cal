# Playback Viewer Model Owner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Viewer model/loading responsibility from the overlay owner without changing playback behavior.

**Architecture:** Keep the existing `playback_viewer.py` composition root and move the already-consumed Viewer helpers into it. Keep `playback_overlay.py` as a pure rendering/selection owner and preserve all current script-facing function names.

**Tech Stack:** Python, MuJoCo, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-31-playback-viewer-model-owner-design.md`

## Global Constraints

- Use `uv run` for Python and test commands.
- Preserve the dirty checkout and all unrelated user changes.
- Do not create a branch, commit, push, train, simulate or run a live Viewer.
- Preserve function names, call order, fallback order, exceptions and log text.

---

### Task 1: Architecture fitness test

**Files:**
- Modify: `tests/visualization/test_playback_deep_owner_boundaries.py`

**Interfaces:**
- Consumes: `unilab.visualization.playback_overlay`, `playback_viewer`
- Produces: a source-level invariant that Viewer model/loading functions are absent from the overlay owner

- [ ] Add a test pinning the owner boundary.
- [ ] Run the test and confirm RED because the overlay currently defines Viewer model loaders.

### Task 2: Move Viewer resource helpers

**Files:**
- Modify: `src/unilab/visualization/playback_overlay.py`
- Modify: `src/unilab/visualization/playback_viewer.py`

**Interfaces:**
- Consumes: current private Viewer helpers and shared render-play resolver
- Produces: identical Viewer-owned `_load_resolved_visual_viewer_model` and `_load_viewer_model` entrypoints

- [ ] Move model parsing/loading, backend/GLFW launch and camera/focus helpers into `playback_viewer.py`.
- [ ] Remove obsolete overlay imports and private reverse traversal.
- [ ] Run the architecture test to confirm GREEN.

### Task 3: Compatibility and regression closure

**Files:**
- Verify: `tests/visualization/`
- Verify: `tests/scripts/test_visualization_entrypoints.py`
- Verify: `scripts/deploy/check_unilab_g1_distill_viewer_path.py`

**Interfaces:**
- Consumes: unchanged script and Viewer entrypoints
- Produces: offline compatibility evidence and final maintainability receipt

- [ ] Run scoped Ruff, format and compile checks.
- [ ] Run focused visualization/script regression, then the broader repository regression.
- [ ] Review the complete diff and validate `FINAL_GATE_PASS` plus the one-shot completion receipt.
