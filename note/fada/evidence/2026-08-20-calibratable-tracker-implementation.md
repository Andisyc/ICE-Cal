# Calibratable Tracker v007/v006 Offline Implementation Evidence

Identity: `codex/in-context-execution-calibration@5949136e43d386a94642a20272e36bd58a53b061`

Content identity: `2f31fc8703d560c92054bb89803283715188356cc175e64685797500d197dcd9`

## Implemented scope

- Config-owned gain, delay, and offset axis catalog plus analytic labels.
- Serial Stage 1 Direction Bank, Stage 2 Coefficient Encoder, and Stage 3 typed scale evidence/PCHIP owners.
- Frozen latent composition, 30-frame cold start, range/jump/reset state, six-Action decode, and first-Action consumption.
- Typed dataset, Stage 3 evidence, checkpoint, deployable artifact, and full-finetune upper-bound persistence.
- Preparation, training, held-out evaluation, and playback composition entrypoints.

## Current evidence

- Module manifest: `note/testing/module_test_manifest.json`, `ADMITTED-OFFLINE`.
- Module suite: 42 passed; normalized stdout SHA256 `a999fb85a52abc1f74bd6a09a59f2706d39c85dba484bef5c3ab616371cd6618`.
- Affected suite: 211 passed, including historical-route regression coverage.
- Ruff check/format and mypy passed for the affected v007 files.
- Final review: `note/fada/reviews/2026-08-20-calibratable-tracker-final-gate.json`, `FINAL_GATE_PASS`.

## Boundary

`formal-runtime-audit` has not run for v007/v006. No simulator collection, long training, live
playback/deployment, convergence, calibration efficacy, robustness, or policy-quality claim is
admitted. Existing v006/v005 technical receipts remain historical.
