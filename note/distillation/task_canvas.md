# Distillation Task Canvas

## Objective

Maintain one resumable multi-role distillation workflow with student-state
DAgger while preserving explicit owner, artifact, checkpoint, and physical
acceptance boundaries.

## Human Decision

Current status: **integration complete, promotion deferred, default off**.

- Active training contract: `DISTILL-TRAIN-v003`.
- Persistent DAgger runtime: integrated and explicit opt-in.
- Default: `training.workflow.execution_mode=legacy`.
- Promotion: deferred because E67 reports `NO_STABLE_SPEEDUP`.
- HP-5: no recurring owner and not authorized.

## Current Evidence

- E34-E40: persistent protocol, barrier, weight publication, resource cache,
  semantic differential, and bounded G1 lifecycle PASS.
- E41-E67: structured timing, formal A/B, two-iteration amortization, repeated
  discriminator, and oracle acceptance complete; verdict
  `NO_STABLE_SPEEDUP`.
- E70-E71: production-readiness and Architecture consistency PASS.
- E86: exact `make test-all` PASS: Ruff/mypy/Pyright green; 1556 passed,
  51 skipped, 256 deselected; 70% coverage.

## Current Owners

- Semantic workflow and lineage: `src/unilab/algos/torch/distill/workflow.py`.
- Persistent process lifecycle: UniLab `AsyncRunner` through
  `distill/async_runtime.py`.
- Student weight versions: UniLab `SharedWeightSync` through
  `distill/persistent_runtime.py`.
- Exact teacher/env resources: `distill/persistent_resources.py` and
  `distill/g1_persistent_worker.py`.
- Structured performance evidence: `distill/performance.py`.
- Default route selection: `conf/distill/config.yaml` and
  `scripts/train_distill.py`.

## Open Boundaries

- RT-10 formal artifact exists, but physical walk-to-stop acceptance is not
  recorded as PASS.
- Persistent execution has no stable end-to-end speedup and remains OFF-default.
- Optional Motrix runtime is unverified in the current environment; provider-
  dependent Stewart tests skip explicitly.
- Slow/S4 tests are outside E86.
- Height teacher checkpoint and promoted student checkpoint have no accepted
  owner.
- The manual collect/offline route is intended as diagnostic-only, but its
  explicit formal labeling remains a checklist item.

## Current Documents

- Concept Figure:
  `note/architecture/concept/03_g1_multiteacher_distillation_method.data.json`
- Active method contract:
  `note/distillation/contracts/active/method/DISTILL-METHOD-v001.md`
- Active training contract:
  `note/distillation/contracts/active/training/DISTILL-TRAIN-v003.md`
- Current acceptance:
  `note/distillation/checklists/current.md`
- Current evidence:
  `note/distillation/evidence/current.md`
- Runtime/owner views: `note/architecture/runtime/` and
  `note/architecture/architecture/`.

## Next Human Decision

No automatic implementation, training, promotion, default-on, commit, or PR
action is active. The next action must be separately selected by the user.
