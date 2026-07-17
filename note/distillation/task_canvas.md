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
- E92: HP-7a server discriminator PASS: `31.8345 s` current versus `1.3357 s`
  cached, `23.8338x`; pool construction owns `93.8%` and all semantic
  differentials pass.
- E93: HP-7b freezes one invocation-local immutable CPU label-pool cache bound
  to the exact loaded dataset; HP-7c remains unauthorized.
- E94: HP-7c1 owner implementation and HP-7c2 formal integration PASS: one
  cache build per invocation, exact RNG/index/count equivalence, `8N` bound,
  301 affected tests, targeted Ruff/mypy/Pyright, and Atlas contracts pass.
- E95: server production-path sentinel PASS: one cache build across 512
  updates, sampled-index digest and final RNG state equal, no training, staging
  `2.1668 s` total and `0.004232 s/update`.

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
  staging bottleneck. HP-7a passes E92: current staging is 31.835 s, cached
  candidate staging is 1.336 s (23.83x), label-pool construction owns 93.8%,
  and semantic differential passes. End-to-end speedup remains unconfirmed and
  this does not reopen default-on promotion.

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
- Bounded workflow freeze:
  `note/distillation/plans/hp7c3_bounded_persistent_workflow_freeze.md`
- Runtime/owner views: `note/architecture/runtime/` and
  `note/architecture/architecture/`.

## Next Human Decision

No automatic training, promotion, default-on, commit, or PR action is active.
HP-7c production-path sentinel passes E95, but HP-7c remains partial. The next
Gate 0 attempt is blocked by E96 before remote reads/writes because the
non-interactive SSH session has no accepted authentication identity. No
freeze/oracle/output exists. Resume only in a user-authenticated SSH session or
with an explicitly provided non-interactive identity; Gate 1 remains closed.
