# Distillation Transaction and Playback Runtime Deep Split Plan

**Goal:** close the five residual long lifecycle functions without semantic changes.

1. Add characterization tests for the new typed lifecycle boundaries and record RED before owners
   exist.
2. Introduce a transition transaction with prepare, label/select, step/advance, append, and
   finalize phases; keep the public collector as composition only.
3. Introduce a FADA transaction with prepare, Oracle/rollout, scenario application, window
   admission, and finalize phases; preserve the existing accumulator and exact ordering.
4. Split dataset merge into local load/validate, concatenate, metadata, and final-build phases.
5. Split DAgger into a typed per-iteration context plus collection, metrics, aggregate/update, and
   atomic commit phases.
6. Split interactive playback into session creation, viewer-resource preparation, viewer loop,
   and cleanup helpers while preserving script-level monkeypatch seams.
7. Run focused tests, the complete affected suite, Ruff, compileall, diff checks, dependency
   searches, AST recount, and final maintainability review.

Constraints: use `uv run`; preserve the dirty checkout; no branch/commit/push, live simulation,
training, dataset collection, schema/config changes, or destructive cleanup.
