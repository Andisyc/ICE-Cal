# FADA Two-Stage Planner-IDM Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace interleaved IDM/Planner updates with separately invocable IDM-pretrain and Planner-only phases connected by a strict completed-IDM checkpoint.

**Architecture:** A small phase owner validates Hydra configuration and provides the only rollout/update selection. The existing workflow remains the composition root, the async lifecycle remains intact, and schema-4 checkpoint metadata seals phase and pretrained-IDM identity. Planner gradients pass through a permanently frozen IDM without IDM optimizer steps.

**Tech Stack:** Python 3.10, PyTorch, Hydra/OmegaConf, pytest, UniLab persistent async runtime.

---

## File structure

- Create `src/unilab/algos/torch/distill/fada_training_phase.py`: legal phase values,
  phase-specific path/source admission, rollout/update selection, and canonical IDM tensor digest.
- Modify `src/unilab/algos/torch/distill/fada_trainer.py`: phase-specific optimizer ownership and byte-identity freeze assertion.
- Modify `src/unilab/algos/torch/distill/fada_checkpoint.py`: schema-4 phase metadata, strict training admission, legacy playback reader.
- Modify `src/unilab/algos/torch/distill/fada_workflow_setup.py`: fail-closed Hydra phase contract.
- Modify `src/unilab/algos/torch/distill/fada_workflow.py`: construct exactly one phase and load only completed pretrained IDM weights for Planner.
- Modify `src/unilab/algos/torch/distill/fada_persistent_workflow.py`: invoke only the phase-owned optimizer and persist phase completion.
- Modify `src/unilab/algos/torch/distill/fada_legacy_workflow.py`: preserve the same phase semantics on the supported legacy route.
- Modify `src/unilab/algos/torch/distill/fada_async_runtime.py`: select Oracle/student rollout from phase rather than iteration alone.
- Modify `conf/distill/config.yaml`: add `phase` and `pretrained_idm_path`; assign fresh v010 output paths.
- Modify focused FADA tests and package exports.

### Task 1: Phase contract and configuration admission

**Files:**
- Create: `src/unilab/algos/torch/distill/fada_training_phase.py`
- Modify: `src/unilab/algos/torch/distill/fada_workflow_setup.py`
- Modify: `conf/distill/config.yaml`
- Test: `tests/algos/test_fada_workflows.py`

- [ ] **Step 1: Write failing phase-admission tests**

Add tests that call the production setup boundary and independently expect:

```python
assert resolve_fada_training_phase(idm_cfg).name == "idm_pretrain"
with pytest.raises(ValueError, match="pretrained_idm_path is required"):
    resolve_fada_training_phase(planner_without_idm_cfg)
with pytest.raises(ValueError, match="must be null"):
    resolve_fada_training_phase(idm_with_pretrained_path_cfg)
with pytest.raises(FileExistsError, match="fresh output"):
    resolve_fada_training_phase(phase_with_existing_output_cfg)
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/algos/test_fada_workflows.py -k 'training_phase' -q`

Expected: collection failure because `resolve_fada_training_phase` does not exist.

- [ ] **Step 3: Implement the minimum phase value object**

Create one frozen `FADATrainingPhase` value with `name`, `updates_idm`, `updates_planner`, and
`main_rollout_uses_student(iteration)`; accept only `idm_pretrain` and `planner`. Add config
validation so only Planner requires `pretrained_idm_path`, and cross-phase half-open states reject.
Resolve and reject an existing output path for both phases before any save or runtime construction.

- [ ] **Step 4: Run GREEN**

Run the exact RED command and require all selected tests to pass.

### Task 2: Phase-specific gradient and optimizer authority

**Files:**
- Modify: `src/unilab/algos/torch/distill/fada_trainer.py`
- Test: `tests/algos/test_fada_planner_idm.py`

- [ ] **Step 1: Write failing semantic optimizer tests**

Use asymmetric parameter snapshots around one update:

```python
idm_before = clone_state(policy.idm)
planner_before = clone_state(policy.planner)
stats = trainer.update_from_replay(...)
assert state_changed(policy.idm, idm_before) is expected_idm_change
assert state_changed(policy.planner, planner_before) is expected_planner_change
```

Cover `idm_pretrain`, `planner`, and a controlled wrong-owner call that must raise. For Planner,
assert all IDM parameters have `requires_grad=False`, `.grad is None`, and state bytes are identical.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/algos/test_fada_planner_idm.py -k 'phase_updates' -q`

Expected: failure because the existing trainer always updates both modules.

- [ ] **Step 3: Implement phase-owned update methods**

Make phase mandatory in `FADATrainer`. Construct only the active optimizer, reject inactive update
counts, freeze IDM once at Planner trainer construction, hold IDM in eval mode, and assert its
canonical digest after every Planner update. The digest uses sorted keys plus dtype, shape, and
contiguous CPU bytes. Return `None` for inactive loss/gradient fields.

- [ ] **Step 4: Run GREEN**

Run the exact RED command, then all `test_fada_planner_idm.py` tests.

### Task 3: Schema-4 phase checkpoint migration

**Files:**
- Modify: `src/unilab/algos/torch/distill/fada_checkpoint.py`
- Modify: `src/unilab/algos/torch/distill/__init__.py`
- Modify: `src/unilab/algos/torch/distill/fada_training.py`
- Test: `tests/algos/test_fada_persistence.py`

- [ ] **Step 1: Write failing old/new matrix tests**

The independent matrix is:

```text
schema 3 -> inference reader: accept
schema 3 -> v010 pretrained-IDM admission: reject
schema 4 incomplete IDM -> Planner admission: reject
schema 4 completed IDM -> Planner admission: load IDM only
schema 4 tampered IDM bytes/hash metadata -> Planner admission: reject before policy mutation
schema 4 -> schema 4 round trip: preserve phase/completion/hash
schema 4 missing/extra optimizer owner -> training load: reject
output path equal to pretrained IDM path -> admission: reject before save
```

Also initialize Planner to a distinct constant before loading and assert it remains unchanged while
IDM becomes equal to the source checkpoint.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/algos/test_fada_persistence.py -k 'phase or pretrained_idm or schema_4' -q`

Expected: failures because schema version is 3 and no phase admission exists.

- [ ] **Step 3: Implement schema-4 owner behavior**

Persist `training_phase`, `phase_completed`, `optimizer_owner`, one `optimizer_state_dict`, and
`pretrained_idm_sha256`. Add a strict `load_pretrained_idm_checkpoint` boundary. Keep
`load_fada_policy_checkpoint` able to read schema 1-4 for playback; reject v010 resume entirely.

- [ ] **Step 4: Run GREEN**

Run the RED command and the complete persistence file.

### Task 4: Phase-correct collection and official workflows

**Files:**
- Modify: `src/unilab/algos/torch/distill/fada_async_runtime.py`
- Modify: `src/unilab/algos/torch/distill/fada_workflow.py`
- Modify: `src/unilab/algos/torch/distill/fada_persistent_workflow.py`
- Modify: `src/unilab/algos/torch/distill/fada_legacy_workflow.py`
- Test: `tests/algos/test_fada_async_worker.py`
- Test: `tests/algos/test_fada_workflows.py`

- [ ] **Step 1: Write failing rollout and workflow tests**

Assert the exact rollout sequences and source families:

```python
assert idm_rollout_modes == ["oracle", "oracle"]
assert planner_rollout_modes == ["oracle", "planner_idm"]
assert planner_intermediate_teacher_loads == 0
```

Also assert that every configured intermediate Oracle contributes in IDM pretraining, preserved
scenario/cold-start quotas and source-role spans remain exact, and Planner replay contains only
main Planner-eligible rows with zero intermediate/planner-ineligible rows.

Use production workflow fakes to assert IDM phase never calls Planner update, Planner phase never
calls IDM update, Planner admission happens before runtime construction, and saved final checkpoint
has `phase_completed=True`.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/algos/test_fada_async_worker.py tests/algos/test_fada_workflows.py -k 'phase or two_stage' -q`

Expected: IDM iteration one incorrectly uses the student and workflows call both updates.

- [ ] **Step 3: Implement phase propagation**

Propagate the resolved phase in the existing cfg payload. Let the phase owner choose main rollout
and whether intermediate sources exist. Planner phase disables paper-source allocations and 1:2
IDM retention so its replay contains only Planner-eligible main rows. At the composition root,
reject resume/warm-start and path aliasing, then load a completed pretrained IDM before
optimizers/runtime/replay. In both
workflow implementations call one phase-owned trainer update and persist completion only after the
last configured iteration.

- [ ] **Step 4: Run GREEN**

Run the exact RED command, then all four focused FADA owner test files.

### Task 5: Configuration, documentation, and affected regression

**Files:**
- Modify: `note/fada/README.md`
- Modify: `note/fada/contracts/README.md`
- Modify: `note/fada/checklists/current.md`
- Modify: active/history v009/v010 contracts
- Create: module/migration/review evidence under `note/fada/evidence/`

- [ ] **Step 1: Compose both official configurations without creating an environment**

Run both phase configurations through `resolve_fada_training_phase` and the production workflow
setup boundary without creating an environment, then run:

```bash
uv run python -c 'from omegaconf import OmegaConf; c=OmegaConf.load("conf/distill/config.yaml"); print(c.training.fada.phase)'
uv run pytest tests/algos/test_fada_workflows.py tests/algos/test_fada_planner_idm.py tests/algos/test_fada_persistence.py tests/algos/test_fada_async_worker.py -q
```

Expected: both phases are admitted with distinct fresh outputs; configured default prints
`idm_pretrain`; all focused tests pass.

- [ ] **Step 2: Run impacted regression and static checks**

Run:

```bash
uv run pytest tests/algos/test_fada_*.py -q
uv run ruff check src/unilab/algos/torch/distill/fada_training_phase.py src/unilab/algos/torch/distill/fada_trainer.py src/unilab/algos/torch/distill/fada_checkpoint.py src/unilab/algos/torch/distill/fada_workflow.py src/unilab/algos/torch/distill/fada_workflow_setup.py src/unilab/algos/torch/distill/fada_persistent_workflow.py src/unilab/algos/torch/distill/fada_legacy_workflow.py src/unilab/algos/torch/distill/fada_async_runtime.py
```

Expected: zero failures and zero Ruff diagnostics.

- [ ] **Step 3: Final review and stop boundary**

Require `code-review-expert` `FINAL_GATE_PASS` with `research-ml` and `migration` profiles. Stop
before simulator or long training. The next human decision is whether to run formal-runtime audit
and authorize the IDM-pretraining command on the server.

## Self-review

- Spec coverage: all six Design Inspector decisions map to Tasks 1-4; evidence/compatibility maps to Task 5.
- Placeholder scan: no deferred implementation or unspecified compatibility behavior remains.
- Type consistency: `FADATrainingPhase`, `training_phase`, `phase_completed`, and
  `pretrained_idm_sha256` use one spelling throughout.
- Non-scope: no observation, network, loss formula, simulator, Oracle, reward, or policy-quality
  behavior changes.
