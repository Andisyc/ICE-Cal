# E92 — HP-7a Server Learner-Staging Discriminator PASS

Date: 2026-07-17
Scope: no-training CUDA microbenchmark over the existing iteration-2 aggregate
dataset from
`/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_persistent_test01`.

## Raw Evidence

- Artifact: `/ssd1/cyx/UniLab/hp7a_iteration2_staging.json`.
- Workload: batch size 512, 512 measured updates, 16 warmup updates, seed 0,
  scenario balancing with quotas `0.50/0.25/0.25`, device `cuda:0`.
- Current staging: `31.834524979814887 s`.
- Cached candidate: `1.3356908820569515 s`.
- Current/candidate ratio: `23.833751811489506`.
- CUDA peak allocated bytes: `622215168`.

## Substage Facts

- Current label-pool construction: `29.86621875874698 s`, approximately
  `93.8%` of current staging time.
- Other current substages: balanced sampling `0.05560810677707195 s`, index
  H2D `0.49740539863705635 s`, Python-label recovery
  `0.6469993107020855 s`, and tensor index-select
  `0.7682934049516916 s`.
- Cached candidate builds the pools once in `0.05716664530336857 s`; its
  remaining substages total approximately `1.2785 s`.
- Sampled indices, label counts, string labels, and tensor batches are all
  exactly equal between current and cached paths; the semantic differential
  reports `pass=true`.

## Verdict

HP-7a is `PASS`. Rebuilding label-index pools from the immutable label tuple on
every learner update is the runtime-confirmed dominant staging owner. The
smallest supported HP-7b design is an owner-local, per-dataset immutable pool
cache with explicit dataset-lifetime invalidation. This evidence does not yet
authorize production implementation, batch-schedule generation, replay-budget
changes, default-on promotion, or an end-to-end speedup claim.

Applying the microbenchmark ratio to the earlier `515.90 s` live staging value
suggests approximately `21.65 s`, but that number is an inference only. A
bounded formal workflow remains required after any implementation.

## Human Decision

Option A is selected: authorize HP-7b design only. Return control before HP-7c
implementation.
