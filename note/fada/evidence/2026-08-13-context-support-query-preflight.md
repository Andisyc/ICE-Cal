---
date: 2026-08-13
evidence_class: runtime-confirmed
method_contract: FADA-CONTEXT-METHOD-v004
training_contract: FADA-CONTEXT-TRAIN-v003
status: preflight-passed-training-not-started
---

# Context Support-Query MuJoCo Preflight

## Identity

- Branch: `codex/in-context-execution-calibration`
- Healthy checkpoint: `/Users/sss9999/locomotion/FADA/planner_idm_v005.pt`
- Checkpoint SHA-256: `d35a32d93b0387e534f6fcdd86b724c44187e308dbca1412435bffe95b6ed90c`
- Task: `sac/g1_walk_flat/mujoco_left_knee_070`
- Fault: actuator index `3`, left knee, multiplier `0.7`
- Command owner config: fixed straight line `[0.4, 0.0, 0.0]`

## Command

```bash
uv run scripts/preflight_fada_context_support_query.py \
  --output /tmp/fada_context_support_query_preflight.json \
  collection.num_envs=1 collection.num_pairs=1 \
  collection.support_length=8 collection.max_reset_pairs=8 \
  collection.artifact_path=/tmp/fada_context_support_query_preflight.pt \
  boundary.optimizer_steps_allowed=false boundary.training_started=false
```

## Observed result

- Status: passed.
- Accepted pairs: `1`; rejected pairs: `0`; independent reset pairs: `1`.
- Support shapes: target future `[1,8,6,98]`, realized state `[1,8,98]`, executed action `[1,8,29]`.
- Query shapes: history `[1,30,98]`, Planner Intent `[1,6,98]`, realized future `[1,6,98]`, action chunk `[1,6,29]`.
- Context output contract: one `delta_z` of shape `[1,128]`.
- Zero-Context first-action MSE: `1.8847433238988742e-05` against required minimum `1e-08`.
- Context gradient norm: `1.7283717170357704e-04`.
- Planner frozen: true; IDM frozen: true; optimizer steps: `0`.

## Evidence boundary

This proves real-MuJoCo fixed-`0.7` data collection, serialization, action-supervision signal, and
gradient connectivity before training. It does not prove that optimization reduces held-out action
MSE or that a trained fixed Context improves the second-rollout trajectory.

## One-step entrypoint smoke test

The production entrypoint was separately exercised with one environment, two pairs, and exactly one
optimizer step, writing only under `/tmp`:

```bash
uv run scripts/train_fada_context_support_query.py \
  device=cpu collection.num_envs=1 collection.num_pairs=2 \
  collection.support_length=8 collection.max_reset_pairs=8 \
  collection.artifact_path=/tmp/fada_context_support_query_smoke_dataset.pt \
  training.batch_size=1 training.steps=1 training.log_interval=1 \
  training.checkpoint_interval=1 \
  training.output_dir=/tmp/fada_context_support_query_smoke \
  boundary.optimizer_steps_allowed=true
```

Observed baseline validation first-action MSE was `1.6266345483018085e-05`; after the single update
it was `1.437894934497308e-05`. The schema-v1 final checkpoint strict-read with step `1`, the healthy
checkpoint SHA, four finite metrics, 12 Context state entries, and optimizer state. This is an
entrypoint/persistence smoke test only, not a training-quality result.

## Final deterministic checks

- `50 passed`: Context Support-Query tests plus Planner-IDM and UniLab FADA regressions.
- Ruff: all selected implementation, script, and test files passed.
- Atlas: viewer import and data contracts passed; desktop and narrow layouts were visually checked.
- `code-review-expert/v1`: `FINAL_GATE_PASS`; no unresolved P0/P1 findings.
