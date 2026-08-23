# Test Inventory

Current v008/v007 evidence maps 3 public owner groups to 12 admitted semantic cases across the
35-file calibration surface. The affected suite contains:

- `tests/algos/test_fada_calibration_collection.py`
- `tests/algos/test_fada_context_support_query.py`
- `tests/algos/test_fada_calibration.py`
- `tests/algos/test_fada_calibration_training.py`
- `tests/algos/test_fada_context_support_query_evaluation.py`
- `tests/algos/test_fada_calibration_evaluation.py`
- `tests/scripts/test_preflight_fada_context_support_query.py`
- `tests/scripts/test_fada_calibration_entrypoints.py`
- `tests/scripts/test_fada_context_official_route.py`

| Required area | Current evidence |
|---|---|
| AxisSpec and dataset | m=1/m=2/m=3 selection, non-catalog order, projection, held-out filtering, typed persistence |
| raw schema and Gateway | active raw v2 single identity; exact read-only raw v1 donor plus full mutation matrix |
| Direction Bank and Stage 1 | variable-width shapes, per-axis gradient/freeze/normalization/gate, v3 publication |
| Coefficient Encoder and Stage 2 | strict v3 predecessor reload, Encoder-only gradient ownership, error gate |
| Scale Curves and Stage 3 | typed evidence v2, canonical scan, zero optimizer, final artifact v2 |
| composition and runtime | fresh reload, same-width order admission, zero identity, named thresholds, first Action |
| evaluation | m=1 early N/A; m=2 held-out role; dataset AxisSpec reaches artifact admission |
| persistence lifecycle | unique atomic publication, lineage checks, old trained schema isolation |

The exact affected command reports `338 passed in 22.25s`. Ruff reports `All checks passed`, mypy
reports no issues in 23 source files, and `git diff --check` is clean. The active surface content
identity is `4daeac6d96b3ae6454d72372e9971867692c276d3d7eee89b8b095eec81d9194`.

This inventory admits offline module behavior only. Formal runtime, simulator execution, training
efficacy, live playback, deployment readiness, and policy quality are not claimed.
