# Source Projection Device-Tolerance Debug-Fix Review Object

## Identity

- Project: ICE-Cal
- Contract: FADA-CONTEXT-METHOD-v008 + FADA-CONTEXT-TRAIN-v007
- Base checkout: `7c2ba26f3fd7771ac0bd273eaa7afd3febd7009b`
- Production owner: `calibration_training/lifecycle.py`

## Proven Runtime Cause

The gain rollout was collected with `--device cuda:0`, while Stage 1 reloads the same checkpoint on
CPU. The stored source checkpoint SHA256 matches the selected checkpoint. CUDA row-wise replay is
exact for both Planner Intent and nominal Action. CPU replay differs by at most `6.29e-4` for Planner
Intent and `3.59e-5` for nominal Action, so the former `rtol=1e-5, atol=1e-6` check produced a false
identity rejection before Stage 1 training.

## Review Surface

- `src/unilab/algos/torch/fada_context/calibration_training/lifecycle.py`
  - SHA256: `0cf679bc19511a4efbe55d9ada385745ca8f874227e4000625e99a038ecaab43`
  - Keeps source identity and architecture checks unchanged.
  - Admits bounded float32 device drift with separate Planner and Action absolute tolerances.
- `tests/algos/test_fada_calibration_training.py`
  - SHA256: `1ef668d26b1cf9c4fed41a1f0fc238b05883e8928235c7c9f1333cb7f4ad91f3`
  - Covers admitted observed drift, material Planner drift, and material Action corruption.

## Verification

- RED: the observed-drift regression failed at the former Planner Intent check.
- Focused GREEN: 3 source-projection tests passed.
- Affected suite: 340 tests passed.
- Ruff: passed for the changed production and test files.
- mypy: passed for the changed production owner.

## Evidence Boundary

This object proves only the bounded offline debug fix and its regression behavior. The server has not
received the patch, Stage 1 has not rerun, and no training-quality or policy-quality claim is made.
