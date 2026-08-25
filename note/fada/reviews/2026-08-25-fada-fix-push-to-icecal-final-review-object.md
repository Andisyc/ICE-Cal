# FADA Fix-Push To ICE-Cal Final Review Object

- Target base: `codex/in-context-execution-calibration@67bf6e1ac9538da5b6ff170fceda2c4a96bc6cd0`
- Donor: `fada-fix-push@6c68d86700a2597d2d307070322015d214affd43`
- Production diff identity: `sha256:826d019d39137752dfa369acde5e2bccaf94de06481dea70ca5abd322e400575`
- Surface: 103 files, 15,143 insertions, 4,115 deletions.
- Preserved: every target `fada_context` and Design Inspector file; Context v009/v008 semantics.
- Replaced: v005 interleaved training and its obsolete test with v010 IDM-pretrain then permanently frozen IDM Planner training.
- Added: source/target collection, adaptation, schema-4 stage transfer, collector/async decomposition, lifecycle guards, configs, contracts, and tests.
- Excluded: `.reasonix`, `standing/model_5000.pt`, giant raw discriminator evidence, training, simulator, commit, push, and remote mutation.

## Current evidence

- `18 passed`: exact v010 semantic pseudo-sample suite.
- `507 passed`: all `tests/algos/test_fada_*.py`.
- `637 passed`: FADA, Context, scripts, backend, and playback impacted sweep.
- `407 passed`: shared distill, script, backend, environment, and visualization sweep.
- `2352 passed, 52 skipped, 4 failed`: whole repository; all four failures are pre-existing missing `note/distillation/plans/formal_dagger_*.spec.json` files absent from both target base and donor.
- Ruff: passed on the complete impacted surface.
- Pyright: zero errors on migrated library and new scripts. `scripts/train_distill.py` retains 33 pre-existing errors on unchanged lines; the migration only removes retired v005 imports from that file.
- Module manifest: `ADMITTED-OFFLINE` for the current target production diff.

No runtime, simulator, training convergence, deployment, or policy-quality claim is made.
