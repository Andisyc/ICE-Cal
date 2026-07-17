# E80 — HP-6b8 G1 Gait-Config Compatibility Repair

Result: **PASS**

## Root cause

E76 replaced a structurally typed accessor with a nominal runtime type check.
That contradicted two established compatibility inputs: lightweight
`SimpleNamespace` observation fixtures and structured owner configs that omit
`reward.gait_constraint`. The adjacent reward-mode accessor already expresses
the intended pattern: missing field uses a disabled default, plain dict is
materialized, and structured config remains duck typed.

## Repair

`_gait_constraint_cfg()` now:

- uses `GaitConstraintConfig()` when the field is absent;
- preserves existing plain-dict materialization and caching;
- preserves structured/fixture proxy objects;
- uses a specific static `cast(GaitConstraintConfig, cfg)` instead of imposing
  a new runtime nominal-type contract.

No observation dimension, gait/reward semantic, or Hydra YAML changed.

## Verification

- Exact ten E79 G1 failures: `10 passed in 0.34s`.
- Targeted mypy: no issues in `joystick.py`.
- Targeted Pyright: 0 errors, 0 warnings.
- Targeted Ruff: pass.
- `git diff --check`: pass.

E80 does not cover the remaining Stewart/docs/CLI failures or the full
repository gate.
