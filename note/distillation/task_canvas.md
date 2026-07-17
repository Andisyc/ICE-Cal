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
- E99 closes the HP-7 staging optimization and bounded live route. Staging is
  now 9.32% of wall time; forward/backward own the measured learner cost.
  End-to-end A/B speedup remains unconfirmed and default-on is not reopened.

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
E99 closes HP-7 implementation and bounded live integration PASS. The frozen
r6 workflow completes one iteration and 12,320 updates with accepted checkpoint,
metrics, and cleanup. Staging is 9.32% of wall time; forward/backward now own
the learner cost. No second run, end-to-end A/B claim, workflow-specific GPU
memory claim, promotion, or default-on action is authorized. Persistent remains
legacy/OFF.

E100 returns control to the main session. HP engineering is complete, but no
formal DAgger run is frozen. The next human decision is checkpoint lineage:
start cleanly from the original parent iteration 3, explicitly promote the r6
sentinel checkpoint, or evaluate r6 first. A selected route still requires a
new workload/output/oracle freeze. Do not reuse the r6 supervisor.

The human has now selected the clean, independent formal-training direction.
`plans/formal_dagger_training_identity.md` freezes the planning boundary:
formal lineage starts from the original completed parent iteration 3; the r6
performance sentinel checkpoint and supervisor are excluded. The current step
is FT-0 no-training identity/oracle materialization. FT-1 real training remains
closed until FT-0 returns a complete accepted freeze and the human separately
authorizes execution.

E101-E103 complete the FT-0 formal-identity owner, thin deploy connector, local
file-level preflight integration, and formal r1 spec. The spec has two added
outer iterations from original
parent iteration 3, update schedule `[12320, 12352]`, total `24672`, and a new
output root. The next boundary is one authenticated server no-training
materialization that recomputes and verifies the schedule. FT-1 remains closed.

E104 moves schedule recomputation inside the one-line materializer using the
real aggregate and production offline replay owner. The only next action is the
authenticated server no-training materialization; FT-1 remains closed.

E105 records r1 Gate 0 as permanently failed with `training_executed=false`.
The real compose route and teacher/dataset environment binding now pass locally.
The only current identity is r2; its no-training materialization is next.
