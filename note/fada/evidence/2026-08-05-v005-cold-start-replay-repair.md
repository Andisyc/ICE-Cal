# v005 Cold-start and Replay Repair Evidence

Date: 2026-08-05

Scope: local engineering closure only. No formal v005 training was launched.

## Causal closure

Observed v004 behavior: the closed-loop Planner diverged before a safe handoff to the standing
Oracle. The collector emitted a training window only after `H+K-1` transitions, while deployment at
reset queried the Planner with the reset observation repeated `H` times and an all-zero action
history. Therefore the earliest contradicted relationship was the Planner's reset-time input support.
Uniform replay further diluted the three main scenarios with intermediate walking-Oracle rows, and
aggregate quality metrics could not expose the failing scenario/profile.

The entailed repair is exact reset-time source construction, persisted row provenance, separate
admissible replay views for IDM and Planner, and scenario/profile-resolved evidence. A future v005
closed-loop failure despite valid per-row identities and exact replay quotas would falsify the claim
that these data-contract gaps were sufficient to recover stability; it would not invalidate that the
v004 source contract lacked deployment cold-start support.

## Implemented contract

- `static_stand` remains 25% of main source data and is split 50/50 between exact cold-start and
  steady-state windows.
- Exact cold-start uses repeated reset observation, zero action history, zero command, and `K` causal
  transitions beginning at reset.
- Artifacts persist `command_scenario`, `planner_eligible`, and `cold_start` for every row.
- IDM samples the complete replay. Planner samples only eligible rows at 50/25/25 and static 50/50.
- Intermediate walking-Oracle data is IDM-only.
- Production checkpoints after iteration zero require finite per-scenario Planner-action MSE and
  separate static cold/steady MSE.
- `initial_weights_path` restores only compatible Planner/IDM parameters; optimizer, replay, cursor,
  counters, and metrics remain fresh.

## Verification

- Focused FADA suite passed after implementation.
- Direct tests cover exact cold-start construction, quota sampling, missing-stratum rejection,
  worker artifact composition, parent row-provenance rejection, and final serializer rejection of
  missing/non-finite metrics.
- Focused FADA/playback/visualization regression passed with `58 passed`; Ruff and Architecture
  Atlas schema checks passed.
- The first real sentinel localized and repaired a device-composition defect: the async builder
  converted `training.device=null` to the invalid string `"None"` instead of the parent's `cpu`
  normalization. A focused null/empty/explicit-device regression now protects that boundary.
- The final real MuJoCo persistent-async sentinel collected 8 main rows at exact `4/2/2` scenario
  quotas, including static `1 cold-start + 1 steady-state`. Oracle-shadow valid fraction was `1.0`,
  done rejection was zero, and the production checkpoint serialized all scenario/cold/steady
  metrics. The sentinel intentionally disabled paper-source expansion because the local workspace
  contains the final walking and standing Oracles but not the 20 intermediate checkpoints.

## Remaining boundary

Formal paper-exact training with the 20 intermediate checkpoints and three-scenario, three-seed
closed-loop evaluation remain separate acceptance stages.
