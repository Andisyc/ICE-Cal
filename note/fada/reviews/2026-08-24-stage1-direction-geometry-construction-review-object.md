# Stage 1 Direction Geometry Construction Review Object

## Frozen Review Unit

- Requested behavior: diagnose whether gain rows admit one shared latent correction direction.
- Preserved behavior: formal Stage 1/2/3, the `0.1` gate, artifact schemas, frozen Planner/Tracker,
  dataset provenance, and server state remain unchanged.
- Review mode: `construction`; profiles: `standard`, `module-boundary`,
  `repository-discipline`, `research-ml`.
- Checkout: `codex/in-context-execution-calibration@eeb2a2f3a91d0162e63375329bd2368b32607946`.
- Plan: `sha256:0f5a66db46a3ba8efee4860483b02325cb52dd9658851462cf90f23575e8ac57`.

## Exact Reviewed Owners

- `direction_geometry.py`: `sha256:43b1db14206122651227ed095b47d13bce8fc464c44c5ef7c5d53b25a17465d0`
- `types.py`: `sha256:68342716ca89685930a024b2e7a53eab7279538f482a206731031cd6730f2e54`
- `calibration_training/__init__.py`: `sha256:da0a970e1bcb4d266060dbadc048a4845c7b8708838b9003b059a0d4424ef30d`
- `fada_context/__init__.py`: `sha256:26d0cd832a2ae461e99f82d907c53bfcf837ac29ead1730cd81cf4b4def5de24`
- `diagnose_fada_calibration_direction_geometry.py`: `sha256:695b5dad1095a8d2df7c88a8f221063102376d9819258b14b5efbc02bf860543`
- `test_fada_calibration_training.py`: `sha256:7222f342fc2e46075e628b26cf0a243c77d49719220c4e7d7798cd3c036ea9bb`
- `test_fada_calibration_entrypoints.py`: `sha256:6014ce053ab1841fd78c7092ac93dcc4a371ec288b3e06bb0d511b9c8d8eccd4`

## Review Result

Verdict: `CONSTRUCTION_PASS`. No P0-P3 actionable finding exists inside the diagnostic diff.

- Responsibility ownership: `direction_geometry.py` owns the analytic question; Stage 1 remains the
  production optimizer owner; the CLI is a Humble Object for loading, provenance, and JSON.
- Public interface: one sealed batch, frozen policy, typed identity, and validated config produce
  immutable typed reports. No artifact or output path enters the owner.
- Dependency direction: the diagnostic reuses source-projection, split, first-action ratio, and Axis
  Spec owners. Formal stages do not depend on the diagnostic.
- State lifecycle: the Decoder pseudoinverse is computed under `no_grad`; no optimizer exists; the
  policy state is verified unchanged and local canonical directions are discarded.
- Tensor provenance: zero coefficients and zero target errors are explicitly excluded and counted;
  the first token and first Action are the only supervised coordinates.
- Diagnostic sufficiency: individual minimum-norm fit separates per-row Decoder reachability from a
  shared-direction fit. Normalized SVD plus signed cosine distinguishes dispersion from opposing
  directions; norm quantiles preserve magnitude mismatch evidence.
- Persistence boundary: the CLI prints one report and exposes no publication route.

## Fresh Evidence

- TDD RED: three owner tests failed on the missing module; two CLI tests failed on the missing
  entrypoint; the analytic refactor then failed while Adam and optimizer flags remained reachable.
- Focused direction-geometry cases: 8 passed.
- Complete impacted FADA calibration/context suite: 359 passed.
- Ruff: passed.
- mypy for the three production owners: passed.
- `git diff --check`: passed.

## External Boundary

Current formal `direction_stage_loss` and Stage 2 Action safety loss still compute MSE over the full
six-Action chunk. The current Design Inspector and the confirmed method behavior supervise only the
physically executed first Action. This pre-existing mismatch is not repaired or hidden by the
diagnostic diff. Consequently, the prior `0.723550`/`0.687611` full-chunk diagnostic is not direct
evidence about the design-aligned first-action Direction geometry.

The server geometry report has not run against `calibration_dataset_gain_v2.pt`. Direction
collinearity, shared-direction fit, scale explainability, Stage 1 admission, runtime connectivity,
and policy quality remain unclaimed.
