# FADA Playback Boundary Refactor Implementation Plan

> **For agentic workers:** Execute inline with TDD. Do not create a branch,
> commit, run simulation, or start training without separate authorization.

**Goal:** Separate nominal FADA playback from training randomization and move
G1 action-trace formatting out of the environment owner.

**Architecture:** Playback uses the existing `BackendAdapter` play-profile seam;
the task YAML owns nominal values, while reset-pose sampling remains in the
locomotion DR owner behind a default-on config flag. G1 diagnostics consume an
immutable snapshot and cannot mutate environment state.

**Tech Stack:** Python 3.10, Hydra/OmegaConf, NumPy, pytest, `uv run`.

**Spec:** `docs/superpowers/specs/2026-08-30-fada-playback-boundary-refactor-design.md`

## Global constraints

- Preserve all training and checkpoint behavior.
- Keep configuration facts in Hydra owners, not scripts.
- Keep backend access in the environment; diagnostics receive values only.
- Use `apply_patch` for edits and `uv run` for Python/test commands.

### Task 1: Nominal playback contract

**Files:**
- Modify: `tests/scripts/test_train_scripts.py`
- Modify: `tests/training/test_training_helpers.py`
- Modify: `tests/envs/locomotion/g1/test_gait_constraint.py`
- Modify: `scripts/play_interactive.py`
- Modify: `src/unilab/visualization/interactive_playback.py`
- Modify: `src/unilab/envs/locomotion/common/domain_rand.py`
- Modify: `src/unilab/envs/locomotion/common/dr_provider.py`
- Modify: `conf/distill/task/g1_walk_flat/mujoco_fada_privileged_planner.yaml`

**Produces:** a resolved play environment with no randomization while training
composition remains byte-for-byte equivalent in meaning.

- [ ] Add config and reset tests for the nominal contract.
- [ ] Run the focused tests and verify the intended assertions fail.
- [ ] Add the default-on reset-pose flag and task-owned play profile.
- [ ] Make interactive composition declare play mode and FADA consume the play
      adapter boundary.
- [ ] Run the focused tests and existing playback/config regression tests.

### Task 2: G1 diagnostic owner

**Files:**
- Create: `src/unilab/envs/locomotion/g1/action_trace.py`
- Modify: `src/unilab/envs/locomotion/g1/joystick.py`
- Test: `tests/envs/locomotion/g1/test_gait_constraint.py`

**Produces:** `G1ActionTraceSnapshot` and `emit_g1_action_trace`, with the
environment retaining only snapshot construction.

- [ ] Use the existing action-trace output test as characterization evidence.
- [ ] Add a focused formatter test if the extracted public boundary needs one.
- [ ] Move environment-variable parsing and formatting into the diagnostic
      module without backend or mutable-env access.
- [ ] Keep the environment method as a narrow snapshot producer.
- [ ] Run the characterization and G1 environment tests.

### Task 3: Verification and review

**Files:** all files changed by Tasks 1-2.

- [ ] Run focused playback, config, reset, trace, and FADA tests.
- [ ] Run `uv run ruff check` on changed Python files and `git diff --check`.
- [ ] Inspect the complete diff for training/config/checkpoint changes.
- [ ] Recount hotspot lines and perform the final `code-review-expert` review.
- [ ] Report unrun live simulation/training as unclaimed evidence.
