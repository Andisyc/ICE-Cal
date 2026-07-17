# FT-0 r1 Compose Owner Repair And r2 Refreeze

Date: 2026-07-17

## Observed r1 Failure

The server r1 materializer stopped before workload observation or training.
Freeze and preflight record `accepted=false`, `training_executed=false`, and
Hydra return code 2. Hydra rejected workflow overrides because CLI-generated
overrides preceded `--cfg/--resolve`.

## Root Cause And Repair

The connector inserted Hydra flags into the public `uv run train` argv. UniLab
CLI then placed generated `task` and workflow-enabled overrides first, leaving
positional overrides on both sides of optional flags. A real compose test also
exposed missing `UNILAB_G1_*` teacher/dataset environment interpolation.

UniLab `build_command()` now retains route/script/generated-override ownership.
The connector places compose flags first in the final script argv and binds
reviewed teacher/dataset paths into the environment shared by compose and the
generated supervisor.

## Identity Decision

r1 is permanently failed and is not deleted, repaired, or rerun. Current:

- `plans/formal_dagger_2round_r2.spec.json`;
- `/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_formal_dagger_2round_20260717_r2`.

Workload remains `[12320, 12352]`, total `24672`, from original parent
iteration 3. FT-1 remains closed.

## Verification

- Real owner-to-Hydra compose: PASS, no training.
- Focused owner/connector/workload/HP-7 regression: 24 passed.
- Ruff and mypy: PASS.

