# Context Multi-Window Training-Boundary Evidence

Date: 2026-08-13

Branch: `codex/in-context-execution-calibration`

Checkout base: `dfaa621e7c5aeda5c1fdc2be88f91b224f0640e6` plus the documented uncommitted
multi-window implementation.

The active `FADA-CONTEXT-METHOD-v005` implementation reached the pre-training boundary without
starting formal optimization.

- Fixed method: left-knee strength `0.7`, command `[0.4, 0, 0]`, `H=30`, `K=6`, `L=60`.
- Pair-window result: 8 accepted independent Support-Query pairs, 26 windows per Query, 208 valid
  windows total, anchors `29..54`.
- Tensor path: Support `[8,60,...]`; Query history `[8,26,30,...]`; realized future
  `[8,26,6,98]`; executed first action `[8,26,29]`; fixed `delta_z [8,128]`.
- Supervision signal: zero-Context all-window first-action MSE
  `2.3850443540140986e-05` versus minimum `1e-08`.
- Gradient ownership: Context gradient norm `1.553474721731618e-04`; Planner and IDM frozen and
  gradient-free.
- Execution boundary: `optimizer_steps=0`, `training_started=false`.
- Verification: 39 focused Context/evaluation/entrypoint tests passed; Ruff passed on the changed
  training path.

Commands:

```bash
UV_CACHE_DIR=/tmp/fada-uv-cache uv run pytest -q \
  tests/algos/test_fada_context_support_query.py \
  tests/algos/test_fada_context_support_query_evaluation.py \
  tests/scripts/test_visualization_entrypoints.py

UV_CACHE_DIR=/tmp/fada-uv-cache uv run python \
  scripts/preflight_fada_context_support_query.py \
  --output /tmp/fada_context_multiwindow_preflight.json \
  collection.num_envs=8 collection.num_pairs=8 collection.max_reset_pairs=4 \
  collection.artifact_path=/tmp/fada_context_multiwindow_preflight_dataset.pt
```

The preflight dataset and JSON report were written under `/tmp` and are not formal training
artifacts. This evidence establishes causal construction, serialization, gradient reachability,
and freeze ownership only. It does not establish held-out action improvement or healthy-trajectory
repair.
