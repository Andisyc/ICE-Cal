# E78 — HP-6b6 Collector Pyright Narrowing

Result: **PASS**

## Boundary and fix

The collector entry contracts already require exactly one transition rollout
policy route and require the selected action mode to materialize actions. E77
showed that Pyright did not retain those cross-branch facts through the loops.
The repair repeats those invariants at the two consumer boundaries with
fail-closed `RuntimeError` checks. It adds no fallback and does not change a
valid collection route.

## Verification

- `uv run pyright .../collector.py`: `0 errors, 0 warnings, 0 informations`.
- `uv run mypy .../collector.py`: `Success: no issues found in 1 source file`.
- `uv run ruff check .../collector.py`: `All checks passed!`.
- Direct collector/persistent/differential contracts: `86 passed, 5 warnings
  in 0.54s`.
- `git diff --check`: pass.

## Boundary

E78 proves static control-flow closure and affected collector contracts. It
does not prove the full repository gate, live training speed, or physical
policy quality. A new exact `make test-all` rerun remains a separate decision.
