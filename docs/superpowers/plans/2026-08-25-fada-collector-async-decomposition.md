# FADA Collector And Async Runtime Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the collector and async runtime hotspots behind compatibility facades without changing behavior.

**Architecture:** Execute two serial, independently testable extractions. First separate collector
contract, IO/shadow, window construction, and transaction coordination. Then separate async
configuration and one-request collection while retaining worker resource ownership in the facade.

**Tech Stack:** Python, PyTorch, NumPy, Hydra/OmegaConf, pytest, Ruff, Pyright.

## Task 1: Freeze Structural And Behavioral Boundaries

- Modify: `tests/algos/test_fada_refactor_boundaries.py`
- Test: existing `tests/algos/test_fada_source_collection.py`
- Test: existing `tests/algos/test_fada_async_worker.py`

- [ ] Add owner-module import and facade object-identity assertions.
- [ ] Add forbidden reverse-import assertions for collector and async owners.
- [ ] Assert `fada_collector.py <= 120` lines and `fada_async_runtime.py <= 500` lines; these
  thresholds apply only after their responsibilities move and do not themselves prove quality.
- [ ] Run the new structure test and observe RED because the owner modules do not exist.
- [ ] Record the existing affected tests as the behavior baseline.

## Task 2: Extract Collector Contract, IO, Windows, And Transaction

- Create: `src/unilab/algos/torch/distill/fada_collection_contract.py`
- Create: `src/unilab/algos/torch/distill/fada_collection_io.py`
- Create: `src/unilab/algos/torch/distill/fada_collection_windows.py`
- Create: `src/unilab/algos/torch/distill/fada_collection_transaction.py`
- Modify: `src/unilab/algos/torch/distill/fada_collector.py`

- [ ] Move dataclasses and `_Transition` without changing fields or defaults.
- [ ] Move observation/action/done/reset and `_oracle_shadow_pair` helpers without changing
  validation, device placement, or transaction restoration.
- [ ] Move window builders and `_concat_batches` without changing any `FADASourceBatch` field.
- [ ] Move `collect_fada_source_windows` to the transaction owner without changing its signature.
- [ ] Re-export public and diagnostic compatibility symbols from `fada_collector.py` by identity.
- [ ] Run structure, source collection, input contract, diagnostics, and two-stage tests.

## Task 3: Extract Async Configuration And Request Collection

- Create: `src/unilab/algos/torch/distill/fada_async_config.py`
- Create: `src/unilab/algos/torch/distill/fada_async_collection.py`
- Modify: `src/unilab/algos/torch/distill/fada_async_runtime.py`

- [ ] Move deterministic allocation/curriculum/config functions and re-export the admitted public
  allocator from the facade.
- [ ] Move source concatenation, collection summaries, cold-start aggregation, and the per-request
  collection/artifact transaction.
- [ ] Keep `PersistentFADACollectorWorker` initialization, dependency resolution, resource cleanup,
  and runtime factory in the facade; delegate `collect()` through explicit resident resources.
- [ ] Run structure, async worker, replay/admission, persistence, workflows, and two-stage tests.

## Task 4: Close The One-Shot Refactor

- Create: `note/fada/evidence/2026-08-25-fada-collector-async-decomposition-module-test.json`
- Create: `note/fada/reviews/2026-08-25-fada-collector-async-decomposition-final-gate.json`
- Update: `note/fada/evidence/2026-08-25-fada-collector-async-decomposition-execution-unit.json`

- [ ] Run all `tests/algos/test_fada_*.py` plus Stage-C/D script tests.
- [ ] Run Ruff on all changed production and test files.
- [ ] Run Pyright on all changed production modules.
- [ ] Run import smoke and `git diff --check`.
- [ ] Validate module, final-review, and complete execution receipts.
- [ ] Stop before simulator, server, training, Git commit, or publication.
