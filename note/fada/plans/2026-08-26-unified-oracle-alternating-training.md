# Unified Oracle Alternating Planner-IDM Implementation Plan

> **Execution:** Inline, test-first, one local reversible transaction. No branch, commit, push,
> simulator, server operation, or training launch is authorized.

**Goal:** Replace scenario-switched SAC Oracle authority with one frozen 98-to-29 policy loaded
from `dagger_iteration_8.pt`, and restore one-iteration `IDM updates -> fixed-IDM Planner updates`.

**Architecture:** `fada_oracle.py` is the single checkpoint-format Gateway: it detects SAC versus
distillation-student payloads on the cold path, validates 98/29 dimensions, freezes the policy, and
returns one tensor-to-tensor module. The FADA collector receives that one module for walk,
static-stand, and walk-to-stand; scenario selection continues to own commands and environment
distribution only. `FADATrainer` owns both optimizers and enforces ordered updates while
`fada_checkpoint.py` owns the new paired-optimizer persistence identity.

**Preserved:** `66/29/3`, `H=30`, `K=6`, source schema 4, scenario and cold-start quotas,
intermediate-Oracle trajectory rows for IDM diversity, final-Oracle labels for Planner, persistent
async lifecycle, and schema 1-4 inference playback.

**Retired:** permanent two-launch IDM/Planner training, scenario-dependent final/standing Teacher
selection, `standing_teacher_checkpoint_path`, and SAC-only wording at the FADA composition root.

## Task 1: Freeze the replacement contract and RED cases

**Files:**
- Create `note/fada/contracts/active/method/FADA-METHOD-v011.md`
- Create `note/fada/contracts/active/training/FADA-TRAIN-v011.md`
- Move v010 method/training contracts to their history locations
- Modify `note/fada/contracts/README.md`
- Test `tests/algos/test_fada_unified_oracle.py`
- Test `tests/algos/test_fada_alternating_training.py`

- [x] Record one final Oracle identity for every main scenario and the ordered update invariant.
- [x] Add a real distilled-checkpoint fixture with asymmetric MoE weights and assert the public
      loader returns a frozen 98-to-29 tensor policy.
- [x] Add malformed, wrong-dimension, and SAC-delegation cases.
- [x] Add a tiny optimizer case proving IDM changes before Planner, Planner changes afterward,
      and IDM is unchanged by the Planner pass.
- [x] Run the focused tests and record expected RED failures caused by missing v011 owners.

## Task 2: Add one Oracle checkpoint Gateway

**Files:**
- Create `src/unilab/algos/torch/distill/fada_oracle.py`
- Modify `src/unilab/algos/torch/distill/playback.py`
- Modify `src/unilab/algos/torch/distill/__init__.py`
- Test `tests/algos/test_fada_unified_oracle.py`

- [x] Implement `load_fada_oracle_policy(path, spec, device)` with one cold-path payload
      discriminator: distillation payloads use `load_distillation_student_policy`; all other
      payloads delegate to the existing strict SAC loader.
- [x] Validate `obs_dim == spec.obs_dim` and `action_dim == spec.action_dim`, set eval mode, and
      disable every parameter gradient before returning.
- [x] Change distillation checkpoint loading to `weights_only=True`; do not add fallback
      deserialization.
- [x] Run the Oracle tests to GREEN.

## Task 3: Collapse scenario-dependent Oracle authority

**Files:**
- Modify `src/unilab/algos/torch/distill/fada_workflow_setup.py`
- Modify `src/unilab/algos/torch/distill/fada_workflow.py`
- Modify `src/unilab/algos/torch/distill/fada_async_runtime.py`
- Modify `src/unilab/algos/torch/distill/fada_async_collection.py`
- Modify `src/unilab/algos/torch/distill/fada_collection_transaction.py`
- Modify `src/unilab/algos/torch/distill/fada_collector.py`
- Modify `src/unilab/algos/torch/distill/fada_legacy_workflow.py`
- Modify `conf/distill/config.yaml`
- Test `tests/algos/test_fada_unified_oracle.py`
- Test affected FADA collection/workflow tests

- [x] Rename the composition dependency from SAC-specific loading to FADA Oracle loading.
- [x] Load one resident final Oracle in the persistent worker; retain the separate static
      environment owner but remove the second policy and checkpoint identity.
- [x] Route all main-source labels and Oracle shadows through the same policy object; keep command
      scenario and environment routing unchanged.
- [x] Remove `standing_teacher_checkpoint_path` and reject stale configuration explicitly rather
      than silently ignoring it.
- [x] Keep intermediate checkpoints restricted to IDM trajectory collection; they never become
      Planner-label authority.
- [x] Run focused collection and workflow tests to GREEN.

## Task 4: Restore serialized alternating optimization

**Files:**
- Modify `src/unilab/algos/torch/distill/fada_trainer.py`
- Modify `src/unilab/algos/torch/distill/fada_workflow.py`
- Modify `src/unilab/algos/torch/distill/fada_workflow_setup.py`
- Modify `src/unilab/algos/torch/distill/fada_persistent_workflow.py`
- Modify `src/unilab/algos/torch/distill/fada_legacy_workflow.py`
- Remove `src/unilab/algos/torch/distill/fada_training_phase.py` after stale-reference closure
- Modify `conf/distill/config.yaml`
- Test `tests/algos/test_fada_alternating_training.py`

- [x] Construct one IDM optimizer and one Planner optimizer in the composition root.
- [x] In each outer iteration, sample and apply all configured IDM updates first, then all Planner
      updates through a temporarily frozen IDM; never update both in one backward/step.
- [x] Use Oracle rollout at iteration zero and current Planner-IDM rollout afterward.
- [x] Collect intermediate-Oracle trajectories only for IDM eligibility while preserving final
      unified-Oracle Planner targets.
- [x] Remove v010 phase/pretrained-IDM configuration and code only after direct callers are migrated.
- [x] Run the ordered-gradient and workflow tests to GREEN.

## Task 5: Persist v011 identity without breaking playback

**Files:**
- Modify `src/unilab/algos/torch/distill/fada_checkpoint.py`
- Modify checkpoint and persistence tests

- [x] Write schema 5 with both optimizer states, `training_schedule=alternating_idm_then_planner`,
      unified Oracle identity in runtime config, policy states, iteration, samples, and metrics.
- [x] Keep schemas 1-4 inference-readable; keep all training resume disabled.
- [x] Reject schema-5 missing either optimizer, wrong schedule, or malformed architecture before
      mutating a target policy.
- [x] Prove schema-5 save/load playback and schema-4 inference compatibility.

## Task 6: Close the offline engineering unit

- [x] Run the two new RED/GREEN suites, affected FADA module tests, refactor-boundary tests, Ruff,
      and Pyright on changed production modules.
- [x] Search for stale `standing_teacher_checkpoint_path`, `load_sac_teacher_policy` at FADA
      composition points, `FADATrainingPhase`, and `pretrained_idm_path` references; retain only
      explicit historical/test fixtures.
- [x] Review the complete diff for responsibility ownership, dependency direction, checkpoint
      compatibility, tensor provenance, gradient authority, and technical-debt delta.
- [x] Record module, migration, execution, and final-gate receipts. Do not claim runtime reachability,
      convergence, walking stability, or policy quality.
