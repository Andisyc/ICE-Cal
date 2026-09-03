# FADA 15-Degree Slope Traversal

This route has four separate boundaries:

1. The existing Planner-IDM checkpoint is the frozen source policy.
2. Stage C collects target-only windows after gait startup and ramp entry.
3. Stage D freezes the Planner and base IDM and trains only IDM Q/V LoRA.
4. Evaluation restores the same complete rollout snapshot for zero-shot and adapted policies.

The slope task uses a 15-degree, 0.8 m-wide ramp. Observation noise, domain
randomization, pushes, action latency, and actuator faults are disabled and
checked before environment construction.

## Stage C: Target Collection

```bash
cd /Users/chengyuxuan/ArtiIntComVis/ICE-Cal
uv run --frozen --no-sync python scripts/collect_fada_target.py \
  --config-name=fada_slope_target \
  collection.policy_checkpoint_path=planner_idm_v022_cpu_limited.pt
```

This publishes `target.pt`, `collection.mp4`, `collection_summary.json`,
and `manifest.json` under `artifacts/fada_target/g1_slope_15_mujoco/`.

## Stage D: LoRA Adaptation

```bash
cd /Users/chengyuxuan/ArtiIntComVis/ICE-Cal
uv run --frozen --no-sync python scripts/adapt_fada_target.py \
  --config-name=fada_slope_adapt \
  adaptation.confirm_train=true
```

## Before/After Evaluation

```bash
cd /Users/chengyuxuan/ArtiIntComVis/ICE-Cal
uv run --frozen --no-sync python scripts/evaluate_fada_slope.py
```

Evaluation writes paired slope videos, ramp-coordinate metrics, and the
enabled-by-default flat-ground regression. Existing output directories are
never overwritten.

Automated tests validate contracts and data flow only. They are not simulator,
training, deployment, or policy-quality evidence.

