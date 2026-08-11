# FADA Playback Migration Review

Date: 2026-08-04

Review mode: `migration_review` (preimplementation and final gate)

Overall assessment: APPROVE

Repository discipline: active (`AGENTS.md`, `FADA-METHOD-v002`, `FADA-TRAIN-v002`)

## Boundary

- Accepted behavior: load the completed Planner-IDM checkpoint and execute the paper-defined first action in `G1WalkFlat` MuJoCo playback.
- Isolation: `--algo fada` is the only new selector. Existing `distill`, HORA, PPO, APPO, and off-policy playback routes remain unchanged.
- Owners: checkpoint construction remains in `fada_training.py`; history/reset/receding-horizon state remains in `fada_playback.py`; viewer lifecycle adaptation remains in `interactive_playback.py`; CLI only selects and composes.
- Non-scope: no target adaptation, LoRA, Oracle-shadow augmentation, policy-quality claim, or generic distill checkpoint compatibility.

## Findings

No P0-P3 findings remain.

The initial config review found that the shared distill root defaults playback to zero actions. The composition root now defaults only `--algo fada` to `interactive.action_mode=policy` while preserving explicit `zero` and `random` overrides.

## Responsibility And Dependency Delta

- Added one inference loader that reconstructs architecture from the checkpoint and uses restricted `weights_only=True` deserialization.
- Added one state owner for observation/action histories and per-row episode reset.
- Added one playback session that transfers environment `done` state and complete task commands to that owner.
- Reused the existing distill task/backend configuration without inheriting its standing-only reset override.
- Added no compatibility fallback and no second command owner.

## Evidence

- Exact local/remote checkpoint SHA-256 match.
- Strict real-checkpoint load: 8 iterations, 524,288 samples, finite `(1,29)` action.
- Focused FADA model/playback and CLI regression tests passed.
- Focused Ruff and Python compile checks passed.
- Real `G1WalkFlat` MuJoCo reset plus one FADA action step passed.

Residual risk: one-step execution proves integration and numerical validity, not locomotion quality or long-horizon stability.
