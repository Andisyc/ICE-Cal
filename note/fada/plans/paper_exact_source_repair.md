# Paper-Exact FADA Source Repair Plan

Status: implementation complete on 2026-08-05; formal v003 training blocked by missing source artifacts

Terminal outcome: UniLab constructs the Appendix B.2 source buffer for IDM and Planner without
state-carrier drift, trains from realized and same-state final-Oracle shadow pairs, adds exactly 20
intermediate-Oracle rollout sources at a 2:1 budget, and records independent source-quality metrics.

Scope: source-domain Planner-IDM construction only. Target adaptation, LoRA, and later FADA stages
remain excluded.

## Step 1 / 2 - Implementation and bounded verification

Status: completed.

Owners: `NpEnv` and MuJoCo snapshot boundaries, FADA collector/batch/loss/trainer/checkpoint,
`train_distill.py`, config, tests, contracts, and Architecture projections.

Acceptance: authoritative all-row reset; exception-safe exact shadow restoration; dual-source IDM
loss; final-Oracle Planner label; 20-checkpoint and 2:1 fail-closed preflight; schema-v2 quality
metrics; focused static/unit/workflow checks; bounded real MuJoCo shadow and quality sentinel.

## Step 2 / 2 - Fresh formal v003 training

Status: blocked before environment creation.

Required input: final Oracle plus exactly 20 unique, architecture-compatible intermediate Oracle
checkpoints from the same training lineage. Current local/remote audit found only `model_5000.pt`.

Execution: populate `training.fada.intermediate_oracle_checkpoint_paths`, keep
`paper_source_enabled=true`, and launch from a fresh checkpoint with `resume_path=null`.

Stop condition: a schema-v2 checkpoint is saved with all source metrics finite, then a separately
declared closed-loop stability threshold is executed. A finite action or successful simulator step
alone is not acceptance.
