# Test Inventory

Current v007/v006 evidence maps 17 real owner/public-boundary rows to 75 admitted semantic case
rows across these test files:

- `tests/algos/test_fada_calibration.py`
- `tests/algos/test_fada_calibration_training.py`
- `tests/algos/test_fada_calibration_evaluation.py`
- `tests/scripts/test_fada_calibration_entrypoints.py`

| Required area | Current evidence |
|---|---|
| axis labels and dataset | owner unit + persistence differentials |
| Direction Bank and isolated Stage 1 | gradient/freeze/normalization/compensation oracles; `direction_frozen` publication |
| Coefficient Encoder and isolated Stage 2 | strict predecessor-path reload, Encoder-only gradient ownership, error gate |
| Scale Curves and isolated Stage 3 | strict coefficient/evidence path reload, canonical scan grid, zero-optimizer proof |
| composition/deployment | zero identity, token/axis covariance, first Action consumer |
| combination evaluation | held-out multi-axis paired comparison |
| persistence lifecycle | dataset, Scale Evidence, stage-artifact-v2 and deployment-artifact C1-C5 rows |
| schema/export isolation | old generic checkpoint APIs absent from the public export surface; legacy envelopes fail closed |

The exact affected command contains 151 passing tests. Its normalized summary is `151 passed\n`
with SHA-256 `fd8a5f106e1d13ad8b4097f59b3f55eed0db660495e96f1901bf0d892e020c32`.
The manifest binds the complete 21-file active surface with content SHA-256
`73535ba14e08111a43aa36a2340b0403b757a78247fe280abf923905500a6978`; the reviewed implementation
change set is separately recorded as 11 files. The one-test serial/independent fresh-reload Action
equivalence is `CONNECTIVITY-ONLY` and is handed to formal audit. No simulator, long-training,
live-playback, deployment-readiness, or policy-quality claim is admitted here.
