# Paper-Exact FADA Source Repair - Final Gate

Review mode: `final_gate_review`

Verdict: APPROVE implementation; formal training remains externally blocked.

Repository discipline: active (`AGENTS.md`, `FADA-METHOD-v003`, `FADA-TRAIN-v003`, Architecture
Atlas, and testing impact rules).

Reviewed boundary: FADA source batch/loss/trainer/checkpoint, collector provenance, `NpEnv` and
MuJoCo/G1 transaction state, Hydra composition, tests, active contracts, and Architecture maps.

## Findings

No open P0-P3 code-discipline finding remains in the reviewed implementation.

Resolved during final gate:

- P1 state reliability: G1 curriculum and derived reward scales initially escaped the generic
  environment snapshot. A task-extension hook now restores them, with owner-level regression.
- P1 lifecycle: intermediate Oracle files were initially strict-loaded only after environment
  creation. All 20 are now strict-loaded on CPU before env/replay mutation.
- P2 ownership: the paper count and 2:1 allocation initially lived in `train_distill.py`. They now
  belong to `build_fada_paper_source_plan()` in the FADA training owner; the script only resolves
  Hydra paths and assembles dependencies.
- P2 persistence: rollout rejection counts initially stayed only in the returned summary. They are
  now included in schema-v2 checkpoint quality metrics.

## Discipline Check

- Ownership: source-plan constants/allocation, batch provenance, transaction state, loss gradients,
  and checkpoint persistence each have one owner.
- Dependency direction: environment uses declared `SimBackend` snapshot methods; backend-private
  MuJoCo arrays do not leak into the collector or script.
- Legacy safety: the FADA route remains default-off; paper-exact mode rejects replay-divergent
  resume and does not accept the v002 checkpoint as v003 training state.
- Research ML: deployable inputs contain no Oracle future/action fields; those fields remain named
  training evidence. Masks, shapes, finite checks, first-action reduction, fixed-IDM Planner
  gradients, and checkpoint schema are independently tested.
- Reliability: reset, snapshot success/exception, termination, command crossing, bounded replay,
  strict source identity, cleanup, and atomic save boundaries are explicit.

## Residual Boundary

Formal v003 training and walking-stability acceptance are not evidenced. The missing 20 intermediate
Oracle checkpoints are an artifact blocker, not a code approval. The known unrelated height-route
message assertion remains outside this review scope.
