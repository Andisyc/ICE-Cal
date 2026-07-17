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
- E87: local `main` merge `06d31ad6` preserves High Speed DAgger plus HP;
 exact merged gate passes with 1578 passed, 30 skipped, 256 deselected.
- E88-E90: the server persistent live run reuses collector PID `1127593`
 across scenarios/iterations with weight versions 1/2/3. Iteration-2 staging
 is 515.90 s and dominates workflow time; source inspection identifies
 per-update full label-pool reconstruction and device-to-CPU label recovery.

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
- The live persistent run confirms the runtime route but exposes a new learner
  staging bottleneck. Individual staging sub-owner costs and attainable speedup
  remain unconfirmed; HP-7a is authorized and its local probe implementation
  passes E91, but the server CUDA discriminator remains pending and does not
  reopen default-on promotion.

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
- Current performance evidence:
  `note/distillation/evidence/2026-07-17-persistent-live-learner-staging-bottleneck.md`
- Candidate optimization plan:
  `note/distillation/plans/dagger_learner_staging_optimization.md`
- Runtime/owner views: `note/architecture/runtime/` and
  `note/architecture/architecture/`.

## Next Human Decision

No automatic production optimization, training, promotion, default-on, commit,
or PR action is active. The next human action is to run the E91 HP-7a probe on
the existing iteration-2 aggregate dataset, preferably after the active GPU is
idle. HP-7a separates label-pool construction, balanced sampling, CPU-to-GPU
index transfer, GPU index-select, and Python-label recovery. Its fastest
falsifier is that cached pools leave staging near 515.90 s. Return with the JSON
artifact before HP-7b.
