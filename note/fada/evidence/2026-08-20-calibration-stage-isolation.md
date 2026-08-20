# Calibration Stage Isolation Offline Evidence

Identity: `codex/in-context-execution-calibration@0e49ea9a5b940fd5e466a63907179ec86e6a1491`

Implementation content identity (21 active files):
`73535ba14e08111a43aa36a2340b0403b757a78247fe280abf923905500a6978`.

## Implemented boundary

- Stage 1 is an independent Direction Bank transaction and does not construct the Coefficient
  Encoder or inspect Scale Evidence.
- Stage 2 exact-byte loads an admitted Stage 1 v2 artifact before constructing its Encoder or Adam,
  then binds the exact parent digest.
- Stage 3 exact-byte loads an admitted Stage 2 artifact and typed Scale Evidence, constructs no
  optimizer, and publishes the existing deployment schema with complete lineage.
- The serial convenience route invokes the same three public transactions through persisted
  artifacts; three standalone training CLIs expose only their own stage inputs.
- The retired generic calibration checkpoint writer/loader has no active production export or
  caller. Same-schema deployment artifacts missing complete lineage fail closed.

## Current evidence

- Affected suite: 151 passed. Normalized summary `151 passed\n` SHA256:
  `fd8a5f106e1d13ad8b4097f59b3f55eed0db660495e96f1901bf0d892e020c32`.
- Stage isolation Module Alignment: 17 owner rows, 75 semantic cases, zero missing cases; the current
  manifest is `note/testing/module_test_manifest.json`.
- Serial-versus-independent persisted Action equivalence: one passing compatibility case, recorded
  as `CONNECTIVITY-ONLY`, not as official-route proof.
- Ruff check and format check passed for 11 changed Python files.
- Mypy passed for `calibration.py` and `calibration_training.py`.
- Final maintainability review:
  `note/fada/reviews/2026-08-20-calibration-stage-isolation-final.json`,
  `FINAL_GATE_PASS`.

## Residual maintainability risks

- `source_tracker_sha256` fingerprints the complete Planner-IDM checkpoint file, so its name is
  narrower than the current conservative identity scope.
- Four training CLIs repeat digest and stage-identity assembly.

## Evidence boundary

No simulator collection, real calibration training, official CLI/process route audit, server
execution, live playback, convergence, calibration efficacy, robustness, policy quality, deployment,
or Git attribution is claimed. Those remain separate formal-runtime, live-authority, and
policy-quality decisions.
