# Source Projection Device-Tolerance Debug-Fix Review Object

## Identity

- Project: ICE-Cal
- Contract: FADA-CONTEXT-METHOD-v008 + FADA-CONTEXT-TRAIN-v007
- Base checkout: `3f35f183169cba30d4eac415849b278c40e0ca82`
- Production owner: `calibration_training/lifecycle.py`

## Proven Runtime Cause

The gain rollout was collected with `--device cuda:0`, while Stage 1 reloads the same checkpoint on
CPU. The stored source checkpoint SHA256 matches the selected checkpoint. CUDA row-wise replay is
exact for both Planner Intent and nominal Action. CPU replay differs by at most `6.29e-4` for Planner
Intent and `3.5736e-4` for nominal Action, with an Action mean absolute difference of `3.5825e-5`.
The first repair mistakenly used the mean-scale Action drift as its bound, so the server passed the
Planner check but still rejected the nominal Action before Stage 1 training.

## Review Surface

- `src/unilab/algos/torch/fada_context/calibration_training/lifecycle.py`
  - SHA256: `8af597e5834ae1dc34007ada77161d42a1ecc8a915ceb207b2598c5f20ebcba9`
  - Keeps source identity and architecture checks unchanged.
  - Admits bounded float32 device drift with separate Planner and Action absolute tolerances.
- `tests/algos/test_fada_calibration_training.py`
  - SHA256: `925625a5a6bb50f8759055ddf55903a0d72e4511bbd30f08f1a0c6fe73b7183e`
  - Covers admitted observed drift, material Planner drift, and material Action corruption.

## Verification

- RED: the corrected observed-drift regression failed at the nominal Action check.
- Focused GREEN: the corrected observed-drift regression passed.
- Directly affected suite: 93 calibration-training tests passed.
- Affected suite: 340 tests passed.
- Ruff: passed for the changed production and test files.
- mypy: passed for the changed production owner.

## Evidence Boundary

This object proves only the bounded offline debug fix and its regression behavior. The server received
the first under-budget repair and reproduced the Action rejection; it has not received this follow-up
patch, Stage 1 has not rerun after it, and no training-quality or policy-quality claim is made.
