---
contract_id: FADA-TRAIN-v009
status: superseded
effective_date: 2026-08-24
superseded_by: FADA-TRAIN-v010
method_contract: FADA-METHOD-v009
scope: fresh persistent-async Stage-B Planner-IDM source training preparation
---

# FADA Source Training Contract v009

The route remains `persistent_async` with final walking/standing Oracles, 20 intermediate walking
Oracles, 1:2 replay retention, scenario quotas `50/25/25`, static/walk cold-start quotas `50/50`,
ordered IDM then fixed-IDM Planner updates, and exact `66/29/3`, `H=30`, `K=6` dimensions.

Source artifacts must be schema 4 and every row must carry an explicit IDM source role. The worker
assigns roles from rollout identity; the parent validates role spans and metadata before replay
mutation. The IDM update selects exactly one matched pair per row as defined by FADA-METHOD-v009.

This is a replacement of the v007 source-loss/persistence contract, not a compatibility fallback.
Existing v007/v007r1 source artifacts and student checkpoints are rejected for the fresh campaign;
resume and warm start remain disabled. Checkpoint output remains schema 3.

Training start requires current module evidence, schema matrix evidence, official offline route
evidence, formal runtime audit, fresh output paths, and separate launch authorization. This contract
does not itself authorize or execute training.
