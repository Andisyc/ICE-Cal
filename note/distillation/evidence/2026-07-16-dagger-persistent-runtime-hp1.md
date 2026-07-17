# DAgger Persistent Runtime HP-1 Evidence

Date: 2026-07-16

## Boundary

This evidence covers only the persistent collector request/result protocol and
UniLab-owned spawned-process lifecycle. It does not cover real G1 environment
creation, teacher inference, dataset throughput, policy quality, Motrix, or the
currently running server job.

## Red Test

Command:

```bash
uv run --active pytest tests/algos/test_distill_async_runtime.py -q
```

Before implementation, collection failed with:

```text
ModuleNotFoundError: No module named 'unilab.algos.torch.distill.async_runtime'
```

## Focused HP-1 Result

Command:

```bash
uv run --active pytest tests/algos/test_distill_async_runtime.py -q
```

Result:

```text
4 passed in 2.19s
```

Confirmed boundaries:

- two sequential requests execute in the same spawned worker process;
- the worker PID differs from the parent PID;
- request/result identity and observed weight version are explicit;
- a worker exception reaches the parent through `AsyncRunner` diagnostics;
- a version mismatch fails closed;
- repeated close reaps the worker and does not leak a live process.

## Impact-Aware Regression

IPC/runtime group:

```bash
uv run --active pytest \
  tests/algos/test_distill_async_runtime.py \
  tests/ipc/test_async_runner.py \
  tests/ipc/test_shared_weight_sync.py \
  tests/ipc/test_rollout_ring_buffer.py \
  tests/ipc/test_memory_budget.py -q
```

Result: `53 passed in 4.44s`.

Distillation/workflow/script group:

```bash
uv run --active pytest \
  tests/algos/test_g1_distillation_contract.py \
  tests/algos/test_distill_workflow.py \
  tests/scripts/test_train_scripts.py -q
```

Result: `280 passed, 8 skipped, 5 warnings in 5.71s`.

Ruff:

```bash
uv run --active ruff check \
  src/unilab/algos/torch/distill/async_runtime.py \
  tests/algos/test_distill_async_runtime.py
```

Result: `All checks passed!`.

Final ordered affected gate (IPC lifecycle before the known script-module
polluter): `341 passed, 5 warnings in 11.24s`.

## Test Isolation Finding

Putting `tests/scripts/test_train_scripts.py` before
`tests/ipc/test_async_runner.py` in one hand-ordered pytest command produced two
spawn pickling failures because `test_run_motrix_rsl_play_loop_uses_render_spacing_and_offset_mode`
constructs `pytest.MonkeyPatch()` directly and does not call `undo()` after
`_train_rsl_rl` removes and reloads all `unilab.*` modules. The IPC group and
the script group each pass independently. This pre-existing test isolation
defect is recorded but is outside HP-1 production scope; no test was skipped or
weakened to obtain the passing results above.

## Decision

HP-1 passes its planned S1/S2 lifecycle gate. HP-2 may design the DAgger outer
barrier adapter. No speedup claim is authorized yet because the real collector
owner and structured timing artifact belong to HP-3 and HP-4.
