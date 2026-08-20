# Test Inventory

Current v007/v006 files contain 42 admitted offline module tests:

- `tests/algos/test_fada_calibration.py`
- `tests/algos/test_fada_calibration_training.py`
- `tests/algos/test_fada_calibration_evaluation.py`
- `tests/scripts/test_fada_calibration_entrypoints.py`

| Required area | Current evidence |
|---|---|
| axis labels and dataset | owner unit + persistence differentials |
| Direction Bank and Stage 1 | gradient/freeze/normalization/compensation oracles |
| Coefficient Encoder and Stage 2 | causal histories, loss ownership, error gate |
| Scale Curves and Stage 3 | monotone fit, range event, zero-optimizer proof |
| composition/deployment | zero identity, token/axis covariance, first Action consumer |
| combination evaluation | held-out multi-axis paired comparison |
| schema isolation | v006/v005 early rejection |
The affected regression suite contains 211 passing tests, including historical Support/Query routes.
Those old tests remain regression characterization, not current semantic evidence. No v007/v006
formal official-route, simulator, live playback, or policy-quality test has been admitted.
