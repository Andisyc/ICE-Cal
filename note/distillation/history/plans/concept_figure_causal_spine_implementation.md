# Concept Figure Causal Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redraw the active G1 distillation Concept Figure as the approved Causal Spine and prove that no connector crosses a non-endpoint block.

**Architecture:** Keep semantic content in the concept JSON and keep rendering generic in `architecture_atlas.html`. Express every connector with explicit side anchors and orthogonal waypoints, then validate those same geometry objects in the atlas contract checker before browser acceptance.

**Tech Stack:** JSON, browser SVG, rough.js, Node.js contract tests, local Atlas server.

---

### Task 1: Geometry Contract Test

**Files:**
- Modify: `note/architecture/auxiliary/atlas_app/check_distillation_atlas.mjs`

- [x] Add helpers that resolve side anchors, build edge point lists, test orthogonal segments, expand non-endpoint rectangles by `connectorClearance`, and detect segment/rectangle intersection.
- [x] Require explicit `fromAnchor` and `toAnchor` on every Concept Figure edge.
- [x] Require the eight approved interactions and reject any edge segment that intersects a non-endpoint block.
- [x] Run `npm run check` and verify the current figure fails on geometry before changing its coordinates.

### Task 2: Causal Spine Data And Renderer

**Files:**
- Modify: `note/architecture/concept/03_g1_multiteacher_distillation_method.data.json`
- Modify: `note/architecture/auxiliary/atlas_app/architecture_atlas.html`

- [x] Place Command Intent, Teacher Policies, Role Data, MoE Student, and Robot Execution on one shared horizontal spine.
- [x] Place Student-State DAgger below Role Data and MoE Student.
- [x] Replace the old nine-edge route with the seven approved interactions, using explicit anchors and orthogonal `via` points for the upper command route and lower DAgger loop.
- [x] Draw main-spine connectors without labels; retain only `路由条件`, `student states`, and `聚合回灌` on non-local routes.
- [x] Add configurable label offset and keep the canvas-colored text halo so labels never sit on their connector stroke.
- [x] Run `npm run check`, `jq empty`, and the modern Node ESM syntax check.

### Task 3: Browser And Documentation Acceptance

**Files:**
- Modify: `note/distillation/evidence/current.md`
- Modify: `note/distillation/task_canvas.md`

- [x] Reload the real Concept Figure through the repository-local Atlas server.
- [x] Inspect the desktop fit-width screenshot for readable text, separated upper/lower routes, and no connector/block collision.
- [x] Confirm all six block titles and the three retained non-local labels are rendered.
- [x] Record geometry-test and browser evidence without changing checkpoint acceptance status.
- [x] Run `git diff --check` and report unrelated `.pt` and `.codegraph` state without modifying it.
