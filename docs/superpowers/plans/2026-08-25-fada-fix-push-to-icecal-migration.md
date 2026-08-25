# FADA Fix-Push To ICE-Cal Migration Plan

**Status:** authorized one-shot local migration; stop before commit or push.

**Donor:** `fada-fix-push@6c68d86700a2597d2d307070322015d214affd43`

**Target:** `codex/in-context-execution-calibration@67bf6e1ac9538da5b6ff170fceda2c4a96bc6cd0`

**Mode:** `REPLACEMENT`, with checkpoint state-schema compatibility as a secondary risk.

## Required outcome

Move the admitted FADA backbone, source collection, target adaptation, two-stage IDM-then-Planner
training, and collector/async decomposition into the existing ICE-Cal branch while preserving the
entire `src/unilab/algos/torch/fada_context` calibration surface and Design Inspector 09.

## Included

- FADA source/observation/causal-window fixes through donor v009.
- Stage-C target collection and Stage-D adaptation owner modules.
- FADA v010 true two-stage training and schema-4 stage transfer.
- Collector and async-runtime behavior-preserving decomposition.
- Required backend lifecycle, persistent async, playback, config, script, and test consumers.
- Active FADA v010 contracts plus target ICE-Cal v009/v008 contracts with an updated prerequisite.

## Excluded

- `.reasonix/`, `standing/model_5000.pt`, machine-local data, checkpoints, and generated raw logs.
- `note/fada/evidence/2026-08-23-v007-coverage-discriminator.json` and other giant raw evidence.
- Donor changes that downgrade, delete, or supersede ICE-Cal context calibration authority.
- Training, simulator, server, commit, push, branch creation, or remote mutation.

## Execution steps

1. Record target baseline and donor-to-target public API map.
2. Apply selected donor tree changes as a squash-style transplant, not a history merge.
3. Preserve all target-only `fada_context`, calibration configs, tests, plans, and Inspector files.
4. Resolve shared FADA model/training/checkpoint/workflow consumers in owner order.
5. Update the base FADA prerequisite in active ICE-Cal contracts without changing calibration math.
6. Run two-stage/source/adaptation/collector tests first, then all FADA and calibration tests.
7. Run Ruff, Pyright, import/compile smoke, stale-reference checks, and diff review.
8. Leave the verified migration uncommitted and unpushed for human inspection.

## Rollback

Until commit, the remote target `origin/codex/in-context-execution-calibration@67bf6e1a` remains the
immutable rollback identity. The donor remains reachable at `fada-fix-push@6c68d867`.
