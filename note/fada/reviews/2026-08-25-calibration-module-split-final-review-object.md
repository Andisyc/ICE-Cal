# ICE-Cal calibration module split — final review object

Review target: the uncommitted behavior-preserving split of
`calibration.py` and `calibration_collection.py` at
`main@5e955870b651da578c34be0ed98c82d01fd5ce3f`.

## Owner map

- `calibration.py` is a 55-line compatibility facade.
- Calibration identity/types, models, readout, artifact IO, and policy composition
  are owned by five modules of 115–272 lines.
- `calibration_collection.py` is a 50-line compatibility facade.
- Gain protocol/types, provenance, artifact IO, and runtime collection are owned
  by four modules of 130–334 lines.
- `collect_gain_calibration_scenario` remains the environment transaction owner;
  `CalibratedFADAPolicy` remains the policy aggregate.

## Preserved boundaries

- Public facade exports retain object identity with their owner definitions.
- Legacy direct test seams (`torch` and `_APPROVED_POINTS`) remain available on
  their historical facade modules without becoming public exports.
- New owner modules do not import either compatibility facade.
- Artifact schemas, validation rules, atomic publication, rollback, tensor shapes,
  policy behavior, and gain collection lifecycle are unchanged.
- The pre-existing private validation dependency in `calibration_training/io.py`
  now imports the public owner helper directly.

## Verification

- Characterization baseline before extraction: 213 tests passed.
- Post-refactor focused and structural suite: 215 tests passed.
- Ruff: all changed production and test files passed.
- mypy: no issues in 12 changed production modules.
- pyright: 0 errors, 0 warnings in 12 changed production modules.
- `git diff --check`: clean.

## Claim boundary

This evidence supports behavior-preserving offline module correctness and improved
ownership. It does not support simulator runtime, training convergence,
calibration efficacy, deployment, or policy-quality claims.
